"""Reward = -MSE(critic_fwd(explanation), gold_activation) on L2-normalized
vectors (so MSE = 2(1-cos)). Set NLA_LOG_MSE_REWARD=1 to use -log(MSE) instead;
GRPO-normalisation makes them near-equivalent in practice.

Called from miles.rollout.rm_hub via --custom-rm-path nla.reward.nla_rm.
sglang_rollout.py:255 fires this per-sample as each generation completes.

The forward runs on the CRITIC TRAINER (GPUs 2,3, FSDP-sharded, idle during
generation) via Ray remote — live weights, no duplicate model, no checkpoint
staleness. RolloutManager.set_critic_handles (train.py:26) stashed the Ray
handles on args before any rollout fires.

Async accumulator: collect samples until --nla-reward-batch-size is hit (or a
50ms timeout for the tail), then dispatch one batched critic_fwd to both critic
ranks. Event loop stays free during the forward (asyncio.to_thread around ray.get)
so later groups' SGLang callbacks fire → generation pipelines with reward compute.
"""

import asyncio
import math
import os

import ray
import torch

from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

from nla.config import load_nla_config
from nla.schema import extract_explanation, normalize_activation


_MSE_EPS = 1e-8
_USE_LOG_MSE_REWARD = bool(int(os.environ.get("NLA_LOG_MSE_REWARD", "0")))
# Under -mse_nrm, 0.0 is the BEST reward (perfect reconstruction) and -2.0 is
# orthogonal. Under -log(MSE), 0.0 corresponds to mse=1 (mid-range). Use the
# orthogonal-equivalent value so a failed extraction is never advantaged.
FAILED_EXTRACTION_REWARD = -math.log(2.0) if _USE_LOG_MSE_REWARD else -2.0
# Flush timeout: originally 50ms for single-sample async_rm path where
# samples arrive fast (per-sample, not per-group) and 50ms catches tail
# stragglers. With --group-rm routing through the accumulator, groups arrive
# staggered over the ~60s generation window — first group lands alone, 50ms
# fires before more arrive, batch-size=256 never kicks in. 5s lets ~10-30
# groups coalesce; adds ≤5s latency in a 100s+ rollout. Override with
# NLA_REWARD_FLUSH_SECS if the stagger pattern differs.
_TAIL_FLUSH_SECONDS = float(os.environ.get("NLA_REWARD_FLUSH_SECS", "5.0"))
# Soft length penalty: r -= SLOPE * max(0, response_length - START). Counters
# the GRPO length ratchet: within-group advantages keep favoring longer
# explanations even after the genuine quality gradient saturates (measured
# ~140 tokens on Qwen3.5), because the per-sample-mean KL loss is structurally
# length-blind. Default OFF (slope 0). Set START at the measured saturation
# point and SLOPE so the penalty dominates the (tiny) post-saturation gain,
# e.g. START=140 SLOPE=0.0025 → −0.15 at a 200-token cap.
_LEN_PENALTY_START = float(os.environ.get("NLA_LEN_PENALTY_START", "140"))
_LEN_PENALTY_SLOPE = float(os.environ.get("NLA_LEN_PENALTY_SLOPE", "0"))


def _length_penalty(sample: Sample) -> float:
    if _LEN_PENALTY_SLOPE <= 0:
        return 0.0
    return _LEN_PENALTY_SLOPE * max(0.0, float(sample.response_length) - _LEN_PENALTY_START)

_TOKENIZER = None
_CFG = None

_pending: list[tuple[Sample, asyncio.Future]] = []
_drain_task: asyncio.Task | None = None


def _lazy_init(args):
    global _TOKENIZER, _CFG
    if _TOKENIZER is not None:
        return
    # Tokenizer and sidecar from the critic's HF dir. FSDP: args.critic_load IS
    # the HF dir. Megatron: critic_load is torch_dist (no tokenizer, no sidecar),
    # so --nla-critic-sidecar-source must point at the FSDP-generated HF dir.
    # Same arg the trainer-side critic uses for its sidecar — single source of truth.
    sidecar_dir = args.nla_critic_sidecar_source or args.critic_load
    _TOKENIZER = load_tokenizer(sidecar_dir, trust_remote_code=True)
    # Megatron critic_fwd passes attention_mask=None (causal-only). With left-pad
    # the last real token attends left to padding → corrupted. Right-pad puts padding
    # after the last real token where causal never reaches. FSDP doesn't care (passes
    # the mask through), so this is a no-op there. Defense-in-depth for older critic
    # checkpoints saved before prepare_critic_checkpoint forced right-pad.
    _TOKENIZER.padding_side = "right"
    _CFG = load_nla_config(sidecar_dir, _TOKENIZER)
    assert _CFG.critic_prompt_template is not None, (
        f"critic sidecar at {sidecar_dir!r} has no critic_prompt_template"
    )


def _prep_batch(samples: list[Sample]):
    """Extract explanations, tokenize, stack golds. Returns (payload, orig_idx)
    for the subset with valid extractions; FAILED ones get the fixed penalty."""
    dump_path = os.environ.get("NLA_ROLLOUT_TEXT_DUMP")
    if dump_path:
        with open(dump_path, "w") as f:
            for i, s in enumerate(samples[:20]):
                f.write(f"=== sample {i} (status={s.status.name}) ===\n{s.response}\n\n")
    prompts, golds, orig_idx = [], [], []
    for i, s in enumerate(samples):
        # Only COMPLETED samples go through the critic. FAILED covers both
        # extraction-miss AND truncated-with-closed-tag (nla_generate.py:282
        # promotes TRUNCATED→FAILED). Without this, trunc-with-tag gets
        # extract_explanation()→succeeds→rwd≈3.63→adv≈+1.2σ, and 77 completed
        # samples at len[140,150) get adv=+0.75 — net length push stays +ve.
        # We can't fix corr=0.099 (longer IS semantically better up to cap),
        # but we can stop paying the TRUNCATEDs that hit the wall.
        if s.status != Sample.Status.COMPLETED:
            continue
        expl = extract_explanation(s.response)
        if expl is not None:
            prompts.append(_CFG.critic_prompt_template.format(explanation=expl))
            golds.append(s.metadata["activation_vector"])
            orig_idx.append(i)
    if not prompts:
        return None, []
    # add_special_tokens=True matches stage0 extractor (extractors.py:131).
    # Gemma needs BOS here; Qwen has bos_token=None (no-op). See sft_critic.py.
    tok = _TOKENIZER(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
    gold = torch.tensor(golds, dtype=torch.float32)  # [B, d]
    return (tok["input_ids"], tok["attention_mask"], gold), orig_idx


def _mse_to_reward(pred: torch.Tensor, gold: torch.Tensor, scale: float) -> list[float]:
    pn = normalize_activation(pred, scale)
    gn = normalize_activation(gold, scale)
    mse = ((pn - gn) ** 2).mean(dim=1)  # [B]
    if _USE_LOG_MSE_REWARD:
        return [-math.log(max(m, _MSE_EPS)) for m in mse.tolist()]
    return (-mse).tolist()


async def _drain(args):
    global _drain_task
    _drain_task = None
    batch, _pending[:] = _pending[:], []
    if not batch:
        return
    # Once `batch` is detached from _pending, any failure below would orphan the
    # awaiting futures (nla_rm callers hang forever on critic OOM / NCCL timeout).
    # Propagate the exception to every unresolved future, then re-raise so the
    # rollout worker itself dies loudly instead of silently stalling.
    try:
        samples = [s for s, _ in batch]
        rewards = [FAILED_EXTRACTION_REWARD] * len(samples)

        payload, orig_idx = _prep_batch(samples)
        if payload is not None:
            ids, mask, gold = payload
            # All critic ranks must participate in FSDP's per-layer all-gather.
            # Dispatch to every handle; results are identical, take rank 0's.
            # to_thread: ray.get blocks but releases GIL → event loop proceeds.
            handles = args._nla_critic_handles
            refs = [h.critic_fwd.remote(ids, mask) for h in handles]
            pred = await asyncio.to_thread(lambda: ray.get(refs)[0])  # [B, d] CPU
            for j, r in zip(orig_idx, _mse_to_reward(pred, gold, _CFG.mse_scale), strict=True):
                rewards[j] = r - _length_penalty(samples[j])

        for (_, fut), r in zip(batch, rewards, strict=True):
            fut.set_result(r)
    except BaseException as exc:
        for _, fut in batch:
            if not fut.done():
                fut.set_exception(exc)
        raise


async def _flush_after_timeout(args):
    await asyncio.sleep(_TAIL_FLUSH_SECONDS)
    if _drain_task is not None:  # still us, nobody drained in the window
        await _drain(args)


async def nla_rm(args, sample_or_samples, **_kwargs):
    global _drain_task
    _lazy_init(args)

    # batched_async_rm path (--group-rm): group of 8 arrives as a list.
    #
    # OLD: dispatch critic_fwd immediately per group → 64 serial ~2s critic_fwd
    # calls (FSDP collective, one-at-a-time across 6 ranks) = ~128s. This was
    # the ACTUAL rollout bottleneck at 27b — not SGLang, not event-loop blocking.
    # Observed 2.44s between group completions = exactly critic_fwd latency.
    # Generation is parallel on different GPUs but reward-via-critic serializes.
    #
    # NEW: route group members through the accumulator. With _TAIL_FLUSH=0.05s,
    # multiple concurrent groups (all finishing their gather at similar times)
    # coalesce into one big critic_fwd batch. 64 groups × 8 = 512 samples could
    # be 1-4 critic_fwd calls instead of 64. Total reward: ~10-20s vs ~128s.
    if isinstance(sample_or_samples, list):
        futs = []
        for s in sample_or_samples:
            fut = asyncio.get_running_loop().create_future()
            _pending.append((s, fut))
            futs.append(fut)
        if len(_pending) >= args.nla_reward_batch_size:
            if _drain_task is not None:
                _drain_task.cancel()
            await _drain(args)
        elif _drain_task is None:
            _drain_task = asyncio.create_task(_flush_after_timeout(args))
        return await asyncio.gather(*futs)

    # per-sample path (default): accumulate across concurrent coroutines.
    fut = asyncio.get_running_loop().create_future()
    _pending.append((sample_or_samples, fut))
    if len(_pending) >= args.nla_reward_batch_size:
        if _drain_task is not None:
            _drain_task.cancel()
        await _drain(args)
    elif _drain_task is None:
        _drain_task = asyncio.create_task(_flush_after_timeout(args))
    return await fut
