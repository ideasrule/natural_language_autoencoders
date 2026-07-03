"""RL rollout via SGLang input_embeds: build embeds, inject scaled activation, generate.

Called from miles.rollout.sglang_rollout.generate_and_rm (line ~230) as:
    sample = await custom_generate_func(args, sample, sampling_params)

SGLang server MUST be launched with --disable-radix-cache (pass
--sglang-disable-radix-cache on the miles CLI). input_embeds and radix
caching don't mix.

The embedding layer is loaded from safetensors (load_embedding_only —
just the one weight tensor, not the full model). NLAFSDPActor.update_weights
dumps a fresh copy to {save}/_nla_rollout_embed.pt after each SGLang sync;
_maybe_reload_embed picks it up. Trainer is idle during rollout so there's
no race — the dump is fresh when we read it.
"""

import asyncio
import base64
import os
import subprocess
import time
from typing import Any

import numpy as np
import logging
import torch

_logger = logging.getLogger(__name__)
from transformers import AutoConfig

from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls
from miles.utils.http_utils import post
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

from nla.arch_adapters import resolve_embed_scale
from nla.config import load_nla_config_from_args
from nla.injection import inject_at_marked_positions
from nla.models import embed_dump_path, load_embedding_only
from nla.schema import MM_ACTIVATION_KEY, MM_CRITIC_TOKENS_KEY, extract_explanation, normalize_activation


_TOKENIZER = None
_CFG = None
_EMBED: torch.nn.Embedding | None = None
_EMBED_SCALE: float = 1.0
_EMBED_MTIME: float = 0.0
_PREFILL_LEAK_PINGED = False

# bf16-base64: ~12MB JSON body → ~2.8MB string. sglang casts to bf16 on
# receipt anyway (schedule_batch hunk in nla_input_embeds.patch), so bf16
# transport is bit-exact end-to-end. Requires nla_input_embeds_b64.patch on
# the engine. Observed: 4096 reqs × 12MB through one router subprocess leaks
# ~65GB/step; smaller bodies = less buffer pressure regardless of root cause.
_BF16_B64_EMBEDS = os.environ.get("NLA_BF16_B64_EMBEDS") == "1"

# sglang_router_rs (rust .abi3.so, forked from RolloutManager, 210 tokio
# threads) leaks ~142GB/step Private_Dirty. Not history_backend (=none), not
# cache_aware tree (round_robin same), not retries — the rust relay path itself
# never frees per-request allocs. Can't fix without rebuilding rust. Bypass:
# round-robin directly to engine URLs (router used once for /workers discovery).
# This REMOVES complexity — router is one process + 210 threads doing urls[i%6].
# Leak ∝ batch × d_model: qwen7b@128 ~640MB/step (never hit), llama70b@512 ~142GB.
_BYPASS_ROUTER = os.environ.get("NLA_BYPASS_ROUTER") == "1"
_ENGINE_URLS: list[str] | None = None
_ENGINE_URLS_LOCK = asyncio.Lock()


def _lazy_init(args):
    global _TOKENIZER, _CFG, _EMBED, _EMBED_SCALE
    if _TOKENIZER is not None:
        return
    _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    # Same helper as train_actor.init → train/infer injection_scale cannot diverge.
    # injection_scale=None is valid (--nla-injection-scale raw → identity pass-through
    # in normalize_activation). train_actor accepts it, so must we.
    cfg, sidecar_source = load_nla_config_from_args(args, _TOKENIZER)
    assert cfg.critic_prompt_template is not None, (
        f"RL rollout needs critic_prompt_template in sidecar ({sidecar_source!r}) "
        f"to build critic tokens for simultaneous AR training."
    )
    assert not getattr(args, "partial_rollout", False), (
        "nla_generate does not support --partial-rollout (always re-tokenizes "
        "from sample.prompt, ignores prior partial response)"
    )
    _CFG = cfg
    # bf16 storage — the bf16→fp32 conversion of 1GB (whole table) was ~1s per
    # reload. Instead convert only the looked-up rows (~125×3584 ≈ 900KB) per
    # request in generate().
    _EMBED = load_embedding_only(args.hf_checkpoint, dtype=torch.bfloat16)
    # load_embedding_only returns a PLAIN nn.Embedding (raw weight lookup).
    # Gemma/T5 archs multiply by √d_model in their embedding forward — training
    # injection sees post-scale (hook after forward), but here we're building
    # embeds manually. Without this, Gemma injection magnitude is off by ~62×.
    # Qwen/Llama: scale=1.0 (no-op). See arch_adapters._SCALED_EMBED_MODEL_TYPES.
    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    _EMBED_SCALE = resolve_embed_scale(hf_config)
    if _EMBED_SCALE != 1.0:
        print(f"[NLA] embed scale ×{_EMBED_SCALE:.2f} "
              f"(model_type={getattr(hf_config, 'model_type', '?')}) — applied to rollout embeds")


_LAST_EMBED_CHECK: float = 0.0


def _maybe_reload_embed(args):
    """Reload the embedding table if a fresher copy is available.

    Megatron: driver pushes the tensor via RolloutManager.set_embed after each
    update_weights; we read args._nla_embed_weight with an object-identity
    check. Zero NFS — stat() on a hard-mounted NFS hangs when async-save's
    bg-write saturates the server.

    FSDP (single-node): NLAFSDPActor.update_weights writes the file to
    {save}/_nla_rollout_embed.pt (tmp+rename); we mtime-poll it. Called once
    per generate() (512×/rollout) — exists()/stat() are NFS syscalls that
    block the event loop, so rate-limit to 1Hz.
    """
    global _EMBED_MTIME, _LAST_EMBED_CHECK
    pushed = getattr(args, "_nla_embed_weight", None)
    if pushed is not None:
        if pushed is _EMBED.weight.data:
            return
        assert pushed.shape == _EMBED.weight.shape, (
            f"pushed embed shape {tuple(pushed.shape)} != cached "
            f"{tuple(_EMBED.weight.shape)}"
        )
        _EMBED.weight.data = pushed
        return
    # FSDP path (single-node, no driver push): mtime-based file reload.
    now = time.monotonic()
    if now - _LAST_EMBED_CHECK < 1.0:
        return
    _LAST_EMBED_CHECK = now
    dump_path = embed_dump_path(args.save)
    if dump_path is None or not dump_path.exists():
        return  # first rollout, before update_weights has fired
    mtime = dump_path.stat().st_mtime
    if mtime == _EMBED_MTIME:
        return
    weight = torch.load(dump_path, map_location="cpu", weights_only=True)
    assert weight.shape == _EMBED.weight.shape, (
        f"dumped embedding shape {tuple(weight.shape)} != cached "
        f"{tuple(_EMBED.weight.shape)} — model vocab/d_model changed mid-run?"
    )
    # Dump is bf16 (train_actor.py writes weight.detach().cpu(), model is bf16).
    # No whole-table conversion — generate() upcasts only the looked-up rows.
    _EMBED.weight.data = weight
    _EMBED_MTIME = mtime


_DEBUG_TIMING = os.environ.get("NLA_DEBUG_TIMING") == "1"


def _prep_payload_sync(args, messages, activation_vector, sampling_params, sample_index):
    """Everything CPU-bound before the HTTP call. Runs in a thread so the
    ~30ms of tokenize + embed + tensor-from-list + inject doesn't block the
    event loop. Without this, 512 coroutines trickle to SGLang at ~30 req/s;
    batch drains to #running-req: 1 between bursts (79 tok/s avg vs 2770 peak).
    With it, all 512 dispatch fast → SGLang stays at max batch."""
    # --apply-chat-template-kwargs (e.g. '{"enable_thinking": false}' for
    # Qwen3/3.5) MUST match what the SFT stage's loss-mask split used —
    # sft_actor.py reads the same arg, so train/rollout prompts cannot diverge.
    prompt_str = _TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **(getattr(args, "apply_chat_template_kwargs", None) or {}),
    )
    # add_special_tokens=False is LOAD-BEARING for Gemma/Llama. The chat-template
    # string already has <bos> baked in; encode(add_special_tokens=True) would
    # prepend a second one → every position shifts by 1 → injection lands wrong.
    # Qwen has bos_token=None so it's a noop there — don't remove thinking it's
    # unnecessary. The sidecar neighbor check catches double-BOS at load time
    # (compute_canonical_neighbors uses one-step tokenize), but only after
    # you've already shipped a broken checkpoint.
    input_ids = _TOKENIZER.encode(prompt_str, add_special_tokens=False)
    ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # [1, T]

    with torch.no_grad():
        # bf16 lookup → fp32 for injection math + numpy transport (no bf16 in numpy).
        # ~125 rows × 3584 ≈ 900KB — trivial vs the 1GB whole-table cast.
        # _EMBED_SCALE handles Gemma/T5 √d scaling (1.0 for Qwen/Llama).
        # sglang casts to model dtype on receipt (patches/nla_input_embeds.patch
        # schedule_batch hunk — added Mar 13 2026 because Gemma's linear kernels
        # strict-check input/weight dtype, where Qwen's autocast).
        embeds = (_EMBED(ids_tensor) * _EMBED_SCALE).float()  # [1, T, d]

    # np.asarray avoids Python-list iteration (5376 elements). torch.from_numpy
    # is zero-copy on the numpy buffer.
    v_np = np.asarray(activation_vector, dtype=np.float32)
    assert np.isfinite(v_np).all(), (
        f"activation_vector has NaN/Inf (sample index={sample_index}). "
        f"Bad stage0 extraction — check the parquet."
    )
    v_raw = torch.from_numpy(v_np).view(1, -1)  # [1, d]
    v_scaled = normalize_activation(v_raw, _CFG.injection_scale)

    # Reuses the pure injection fn — asserts exactly 1 match with correct
    # neighbors. On miss, AssertionError here is LOUD (not graceful) — if
    # the prompt template drifted this badly, the whole RL run is garbage
    # and should crash, not silently train on ㊗-as-text.
    embeds_injected = inject_at_marked_positions(
        ids_tensor, embeds, v_scaled,
        _CFG.injection_token_id, _CFG.injection_left_neighbor_id,
        _CFG.injection_right_neighbor_id,
    )  # [1, T, d]

    payload, halt_status = compute_request_payload(
        args, input_ids=input_ids, sampling_params=sampling_params
    )
    embeds_out: np.ndarray | tuple[str, list[int]] | None = None
    if payload is not None:
        flat = embeds_injected[0].contiguous()
        # bf16 transport is bit-exact end-to-end (sglang casts to bf16 anyway)
        # ONLY when injection_scale is small enough that the injected vector's
        # magnitude fits bf16's 8-bit mantissa without resolution loss. gemma27b
        # at INJ_SCALE=60000: ~256 resolution → 4% KL spikes (train↔rollout
        # mismatch). llama70b at scale=30: fine. Gate on scale<1000 (heuristic).
        # Env var doesn't reach RolloutManager — use args + cfg.
        scale = _CFG.injection_scale or 0.0
        bf16_safe = getattr(args, "sglang_disable_radix_cache", False) and scale < 1000
        if _BF16_B64_EMBEDS or bf16_safe:
            # numpy has no bf16; view as int16 for transport. Server reinterprets.
            buf = flat.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
            embeds_out = (base64.b64encode(buf).decode("ascii"), list(flat.shape))
        else:
            embeds_out = flat.numpy()
    return input_ids, v_raw, embeds_out, payload, halt_status


async def _resolve_url(args, sample_index: int) -> str:
    # Bypass when radix is disabled — that's the input_embeds signal (radix
    # keys on token IDs and would silently cache-hit across DIFFERENT injected
    # activations). args is in RolloutManager's process; env vars aren't.
    bypass = _BYPASS_ROUTER or getattr(args, "sglang_disable_radix_cache", False)
    if not bypass:
        return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    global _ENGINE_URLS
    if _ENGINE_URLS is None:
        async with _ENGINE_URLS_LOCK:
            if _ENGINE_URLS is None:
                urls = await get_worker_urls(args)
                assert urls, "router /workers returned empty — engines not registered yet"
                _ENGINE_URLS = urls
                _logger.info(f"[NLA] bypassing router, {len(urls)} engines: {urls}")
    return f"{_ENGINE_URLS[sample_index % len(_ENGINE_URLS)]}/generate"


async def generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    _t = time.perf_counter() if _DEBUG_TIMING else 0.0
    _lazy_init(args)
    _maybe_reload_embed(args)
    url = await _resolve_url(args, sample.index)

    messages = sample.prompt
    assert isinstance(messages, list), (
        f"nla_generate requires list[dict] prompt (got {type(messages).__name__}). "
        f"NLADataSource must use apply_chat_template=False; ㊗ substitution "
        f"happens there."
    )

    input_ids, v_raw, embeds_out, payload, halt_status = await asyncio.to_thread(
        _prep_payload_sync, args, messages, sample.metadata["activation_vector"],
        sampling_params, sample.index,
    )
    if payload is None:
        sample.status = halt_status
        return sample
    # compute_request_payload sets payload["input_ids"] — SGLang receives BOTH
    # that and our input_embeds. With both present, SGLang may use input_ids
    # for logprob bookkeeping (logprob_start_len, origin_input_ids slicing)
    # while using input_embeds for the forward — any length/offset mismatch
    # between the two causes train↔rollout logprob misalignment (k3=0.15 floor
    # even at inj=0). Drop input_ids so SGLang has ONLY input_embeds. We still
    # pass input_ids to update_sample_from_response separately (below).
    del payload["input_ids"]
    if isinstance(embeds_out, tuple):
        payload["input_embeds_b64_bf16"], payload["input_embeds_shape"] = embeds_out
    else:
        # orjson OPT_SERIALIZE_NUMPY reads the fp32 buffer in Rust, no
        # Python-float intermediate.
        payload["input_embeds"] = embeds_out

    if _DEBUG_TIMING:
        _t_pre = time.perf_counter() - _t
        _t = time.perf_counter()
    output = await post(url, payload)
    assert isinstance(output, dict) and "meta_info" in output, (
        f"SGLang rejected request: {output!r}. If the error mentions 'Either "
        f"text, input_ids or input_embeds should be provided', your SGLang server "
        f"is missing the NLA transport patches — run patches/apply_sglang_patches.sh "
        f"against your SGLang checkout and restart the server."
    )

    # sglang retract-logprob leak: our nla_sglang_input_embeds_retract_fix.patch
    # (PR #14110) clears Req.output_ids on KV retraction but the scheduler→
    # tokenizer_manager IPC already sent pre-retraction logprobs to tokenizer's
    # state accumulator (tokenizer_manager.py:1665). Retraction is scheduler-
    # internal; tokenizer never knows. Stale pre-retract + fresh post-restart =
    # 200-263 entries for a 150-tok gen. Qwen/12b never retract (KV headroom).
    #
    # Dropping the sample gives a 0-response-length hole in the batch
    # → NCCL timeout at thd packing. Extraction-miss FAILED samples pass miles
    # fine because update_sample_from_response fills real-shaped fields; ours
    # don't. Letting it through unchanged is tolerable (~0.1% of gradient on
    # stale decode logprobs, learned OK). Warn so we track the rate; optionally
    # set NLA_PINGME_CMD=<executable> for a slack/page on first occurrence.
    # Long-term fix: avoid retraction (--sglang-mem-fraction-static) or a
    # scheduler→tokenizer reset signal.
    _lps = output["meta_info"].get("output_token_logprobs")
    _max = payload["sampling_params"].get("max_new_tokens")
    if _lps and _max is not None and len(_lps) > _max:
        global _PREFILL_LEAK_PINGED
        _fin = output["meta_info"].get("finish_reason")
        _logger.warning(
            f"sglang retract-logprob leak: idx={sample.index} "
            f"recv={len(_lps)} > max_new={_max} finish={_fin}. "
            f"Letting through (drop → NCCL timeout). grep 'Retract requests' for rate."
        )
        if not _PREFILL_LEAK_PINGED:
            _PREFILL_LEAK_PINGED = True
            _msg = (f"sglang retract-logprob leak: recv={len(_lps)} > max_new={_max}. "
                    f"Letting through (drop path NCCL-timeouts). Rate = grep -c "
                    f"'Retract requests'. Avoid via --sglang-mem-fraction-static.")
            _cmd = os.environ.get("NLA_PINGME_CMD")
            if _cmd:
                subprocess.Popen(
                    [_cmd, _msg],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                _logger.error(_msg)
    if _DEBUG_TIMING:
        print(f"[nla_timing] idx={sample.index} prep={_t_pre*1000:.0f}ms post={(time.perf_counter()-_t)*1000:.0f}ms", flush=True)
    # generate_endpoint_utils.py:78-82 falls back to EMPTY lists if
    # output_token_logprobs is missing — GRPO would silently train on zero
    # logprobs. Assert loud here; input_embeds is an uncommon SGLang path.
    assert output["meta_info"].get("output_token_logprobs"), (
        f"SGLang returned no output_token_logprobs (input_embeds path). "
        f"Check --sglang-disable-radix-cache is set and SGLang version "
        f"supports logprob with input_embeds. meta_info keys: "
        f"{list(output['meta_info'].keys())!r}"
    )
    # input_ids passed separately so update_sample_from_response can set
    # sample.tokens (generate_endpoint_utils.py:68) — SGLang used the embeds.
    await update_sample_from_response(
        args, sample, payload={"input_ids": input_ids}, output=output
    )

    # Stash RAW activation for both actor training (scaled in
    # _get_model_inputs_args) and critic training (scaled per mse_scale).
    sample.multimodal_train_inputs = {MM_ACTIVATION_KEY: v_raw}

    explanation = extract_explanation(sample.response)

    # Truncated-with-valid-tag gets FAILED too — drops it from critic training
    # (_swap_rollout_to_critic_tokens filters on MM_CRITIC_TOKENS_KEY absent)
    # AND reward.py:_prep_batch gives it rwd=0 → adv≈-2.5. Without this,
    # trunc@150 adv=+1.2 vs completed adv=+0.02 → length drift +0.30 tok/step
    # → FVE peaks after ~30 steps then drops (observed in an early RL run).
    if explanation is None or sample.status == Sample.Status.TRUNCATED:
        sample.status = Sample.Status.FAILED
        return sample

    critic_prompt = _CFG.critic_prompt_template.format(explanation=explanation)
    # add_special_tokens=True — critic_prompt_template is a raw string (not
    # chat-template-processed), so we need BOS here for Gemma to match the
    # stage0 extractor's regime. Contrast _prep_payload_sync above: there the
    # chat template already bakes in BOS, so add_special_tokens=False is correct.
    critic_tokens = _TOKENIZER(critic_prompt, add_special_tokens=True)["input_ids"]
    # Critic is a truncated K+1-layer model with short-context positional
    # embeddings. Long explanation → critic OOM or position-embed index error.
    # Drop (like extraction miss) rather than crash — actor still trains.
    max_ctx = args.rollout_max_context_len or 512
    if len(critic_tokens) > max_ctx:
        sample.status = Sample.Status.FAILED
        return sample
    sample.multimodal_train_inputs[MM_CRITIC_TOKENS_KEY] = torch.tensor(
        critic_tokens, dtype=torch.long
    )

    return sample
