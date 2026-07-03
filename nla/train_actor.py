"""NLAFSDPActor: FSDP training actor with model-type-dispatched NLA support.

Two orthogonal dimensions:
  - model type: LM (.logits) vs NLACriticModel (.values) — gated by _is_critic_model
  - role: "actor" (rollout_data as-is) vs "critic" (swap actor tokens → critic tokens)

Key simplification: override _compute_log_prob to no-op for critic models (they
have no .logits). Stock _train_core then works — compute_advantages_and_returns
early-returns when log_probs/values are None, _train_step override handles .values.
Override _train_core (not train) so parent handles get_rollout_data + timers + perf log.
"""

import os
import threading
import shutil
import subprocess
import sys

import ray

from miles.utils.ray_utils import Box
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

from miles.backends.experimental.fsdp_utils.actor import FSDPTrainRayActor, apply_fsdp2
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.data import get_batch
from miles.backends.training_utils.log_utils import aggregate_forward_results
from miles.backends.training_utils.loss import get_log_probs_and_entropy
from miles.utils.timer import timer
from tqdm import tqdm
from miles.backends.training_utils.loss import loss_function

from nla.arch_adapters import resolve_sglang_key_prefix, resolve_text_config, resolve_text_model
from nla.config import NLAConfig, load_nla_config_from_args, write_model_sidecar
from nla.injection import inject_at_marked_positions
from nla.models import NLACriticModel, embed_dump_path
from nla.schema import (
    MM_ACTIVATION_KEY, MM_CRITIC_TOKENS_KEY, MM_MSE_SCALE_KEY,
    load_predict_mean_baselines, normalize_activation,
)
from nla.storage import _load_storage, is_remote


# === NLA fp32 monkeypatch (keeps Miles unmodified) ===
# NLA_FP32=1 forces fp32 *compute* for FSDP training. bf16 produced NaN grads in
# the first few NLA SFT steps (finite forward, non-finite backward). Miles'
# apply_fsdp2 hardcodes MixedPrecisionPolicy(param_dtype=bf16) and exposes no
# fp32 flag, and the precision lives in base-Miles code NLAFSDPActor inherits via
# super().init() (no override seam) — so instead of editing Miles we patch the
# policy class apply_fsdp2 imports function-locally. param_dtype=fp32 gives fp32
# forward/backward (the part that fixes the NaN); reduce_dtype/storage are
# untouched. Runs at import (Miles imports this module via --custom-actor-cls-path,
# before init/apply_fsdp2). Requires an fp32-capable attention (sdpa) — FlashAttn
# rejects fp32. Process-global, so it also makes the RL KL ref model fp32-compute
# (harmless). No effect unless NLA_FP32=1.
def _nla_maybe_patch_fp32():
    """NLA_FP32=1 -> full fp32 training (storage + compute) without editing Miles.

    bf16 produced NaN grads in the first few NLA SFT steps (finite forward,
    non-finite backward). Full fp32 fixes it; it needs BOTH halves, matching the
    two spots a Miles edit would touch:

      (a) fp32 COMPUTE: Miles' apply_fsdp2 hardcodes
          MixedPrecisionPolicy(param_dtype=bf16). We patch the policy class it
          imports function-locally from torch.distributed.fsdp -> param_dtype=fp32
          (no downcast of the fp32 weights during forward/backward).
      (b) fp32 STORAGE: Miles' base init loads the model with
          from_pretrained(torch_dtype=bf16). Compute-fp32 alone is NOT enough —
          params outside the FSDP wrap list (e.g. NLACriticModel.value_head) never
          get the policy's cast and stay bf16 -> still NaN. So we also force
          torch_dtype=fp32 on the NLA model loaders (our own classes, not Miles).

    MUST run in the worker process (apply_fsdp2 / from_pretrained run there), so
    this is called from NLAFSDPActor.init() — a module-level patch only runs in the
    driver, since Ray ships the actor class to workers by pickling, not by
    re-executing the module. Idempotent. NLA_FP32 needs an fp32-capable attention
    (sdpa); FlashAttention rejects fp32. No effect unless NLA_FP32=1.
    """
    if os.environ.get("NLA_FP32") != "1":
        return

    # (a) fp32 compute — patch the FSDP MixedPrecision policy.
    import torch.distributed.fsdp as _nla_tdf

    if not getattr(_nla_tdf.MixedPrecisionPolicy, "_nla_fp32_patched", False):
        _orig_mp = _nla_tdf.MixedPrecisionPolicy

        def _fp32_mp(*args, **kwargs):
            kwargs["param_dtype"] = torch.float32
            return _orig_mp(*args, **kwargs)

        _fp32_mp._nla_fp32_patched = True
        _nla_tdf.MixedPrecisionPolicy = _fp32_mp

    # (b) fp32 storage — force torch_dtype=fp32 on the NLA model loaders. These are
    # our own classes; Miles' base init calls cls.from_pretrained(torch_dtype=bf16),
    # and we override that kwarg.
    def _make_fp32_loader(orig_func):
        def _fp32_from_pretrained(cls, *args, **kwargs):
            kwargs["torch_dtype"] = torch.float32
            return orig_func(cls, *args, **kwargs)
        _fp32_from_pretrained._nla_fp32_patched = True
        return classmethod(_fp32_from_pretrained)

    for _cls in (NLATextOnlyCausalLM, NLACriticModel):
        _fp = _cls.__dict__.get("from_pretrained")
        if _fp is not None and not getattr(_fp.__func__, "_nla_fp32_patched", False):
            _cls.from_pretrained = _make_fp32_loader(_fp.__func__)

    print("[NLA] NLA_FP32=1: fp32 storage (from_pretrained) + fp32 compute "
          "(MixedPrecisionPolicy) forced", flush=True)
# === end NLA fp32 monkeypatch ===


CRITIC_ONLY_MM_KEYS = {MM_CRITIC_TOKENS_KEY}


def _swap_rollout_to_critic_tokens(rollout_data: dict, device: torch.device) -> dict:
    """Rewire rollout_data: actor tokens → critic tokens, filter failed extractions.

    Pure data transform — unit-testable. The RL rollout fn stashed
    `nla_critic_tokens` (tokenized <text>{payload}</text> <summary>{pm}) in
    multimodal_train_inputs for samples where <explanation> extraction succeeded.
    Missing key → sample filtered.

    Returns a NEW dict. Caller must handle cross-rank divergence — len(kept) may
    differ per rank; see _truncate_to_cross_rank_min.
    """
    kept: list[int] = []
    critic_tokens: list[torch.Tensor] = []
    mm_list = rollout_data["multimodal_train_inputs"]
    for i, mm in enumerate(mm_list):
        if mm is None or MM_CRITIC_TOKENS_KEY not in mm:
            continue
        critic_tokens.append(mm[MM_CRITIC_TOKENS_KEY])
        kept.append(i)
    # No assert on len(kept) here — it's a PER-RANK check. If one rank has zero
    # and asserts, the others enter _truncate_to_cross_rank_min's all_reduce and
    # hang forever. Let the collective assert n_min > 0 fire on ALL ranks together.
    empty_mask = torch.empty(0, dtype=torch.int, device=device)
    return {
        "tokens": critic_tokens,
        "total_lengths": [t.shape[0] for t in critic_tokens],
        "response_lengths": [0] * len(critic_tokens),
        "loss_masks": [empty_mask] * len(critic_tokens),
        "multimodal_train_inputs": [
            {MM_ACTIVATION_KEY: mm_list[i][MM_ACTIVATION_KEY]} for i in kept
        ],
    }


def _assert_reward_train_paths_agree(
    critic_fwd_fn, model: torch.nn.Module, rollout_data: dict, mse_scale: float,
    tol: float = 0.10, qkv_format: str = "thd",
) -> None:
    """Live step-0 check: padded critic_fwd MSE == thd-packed training MSE.

    Preflight (rl_preflight.py) validates this on dummy data with the HF-loaded
    critic. This runs it once at step 0 on REAL rollout data with the REAL
    post-DCP-overlay critic — catches anything the dummy batch misses (e.g. DCP
    load corrupted weights, or a tokenizer edge case the dummies don't hit).

    The old the left-pad fix bug (left-pad + mask.sum-1) would give per-sample ratios
    ~1.5-2.0 here. bf16 GEMM tiling noise is ~1e-4.
    """
    mm_list = rollout_data["multimodal_train_inputs"]
    toks = [mm[MM_CRITIC_TOKENS_KEY] for mm in mm_list if mm and MM_CRITIC_TOKENS_KEY in mm]
    golds = torch.cat([mm[MM_ACTIVATION_KEY] for mm in mm_list if mm and MM_CRITIC_TOKENS_KEY in mm], dim=0)
    # Two paths must agree on ANY subset — a handful of varied-length samples
    # exercises padding edge cases; 32 from the rank-partition adds ~1s.
    # critic_fwd returns .cpu() but rollout_data's golds are on the rank's
    # CUDA device (miles moved them during data prep). Unify on CPU.
    toks, golds = toks[:32], golds[:32].float().cpu()
    n = len(toks)
    if n < 4:
        print(f"[NLA STEP0 CHECK] skipped: n={n} < 4 (smoke-test batch too small for varied-length padding)", flush=True)
        return

    # Reward path: pad to max, attention_mask, critic_fwd picks last_idx.
    lens = torch.tensor([t.shape[0] for t in toks])
    T = int(lens.max())
    pad_id = 0  # never attended — last_idx picks before padding
    ids = torch.full((n, T), pad_id, dtype=toks[0].dtype)
    mask = torch.zeros((n, T), dtype=torch.long)
    for i, t in enumerate(toks):
        ids[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = 1
    pred_reward = critic_fwd_fn(ids, mask)  # [n, d] CPU

    # Training path. bshd (hybrid linear-attention models): right-padded rows,
    # per-row position_ids, mask=None — exactly what get_batch builds. thd:
    # concat + position_ids reset at boundaries (packed-detection gate).
    if qkv_format == "bshd":
        pos = torch.zeros((n, T), dtype=torch.long)
        for i, l in enumerate(lens.tolist()):
            pos[i, :l] = torch.arange(l)
        with torch.no_grad():
            values = model(
                input_ids=ids.cuda(), position_ids=pos.cuda(),
                attention_mask=None, use_cache=False,
            ).values
            pred_train = values[torch.arange(n).cuda(), (lens - 1).cuda()].float().cpu()
    else:
        packed = torch.cat(toks).unsqueeze(0).cuda()
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), lens[:-1].cumsum(0)])
        pos_ids = torch.cat([torch.arange(int(l)) for l in lens]).unsqueeze(0).cuda()
        with torch.no_grad():
            values = model(input_ids=packed, position_ids=pos_ids, attention_mask=None, use_cache=False).values
            pred_train = values[0, (offsets + lens - 1).cuda()].float().cpu()

    def _mse(p: torch.Tensor) -> torch.Tensor:
        pn = normalize_activation(p, mse_scale)
        gn = normalize_activation(golds.float(), mse_scale)
        return ((pn - gn) ** 2).mean(dim=1)

    r = (_mse(pred_reward) / _mse(pred_train)).numpy()
    dev = abs(r - 1.0).max()
    print(f"[NLA STEP0 CHECK] reward/train MSE ratio: mean={r.mean():.4f} max|r-1|={dev:.4f} n={n}", flush=True)
    assert dev < tol, (
        f"step-0 reward-path and training-path MSE diverge by {dev:.1%} (tol {tol:.0%}) on real "
        f"rollout data. Preflight passed — either DCP overlay corrupted the critic, or these "
        f"tokens hit an edge case the dummy prompts missed. Per-sample ratios: {r}"
    )

    # NOT checking raw pred_norm/gold_norm: normalize_activation(v,s) does
    # v/‖v‖·s — MSE loss is scale-invariant, head output norm is unconstrained.
    # Gemma's head naturally outputs at backbone scale (~3× gold). Preflight's
    # normalize(pred).norm/√d > 0.1 is the right check for random-direction
    # (Mar 13 bug); this step-0 check covers path-divergence only.


def _truncate_to_cross_rank_min(
    rollout_data: dict, dp_group, micro_batch_size: int | None
) -> dict:
    """All-reduce len(tokens) to the cross-rank MIN and truncate all lists.

    After _swap_rollout_to_critic_tokens, each rank may have a different
    len(kept). get_data_iterator computes num_steps = len(tokens) // (gbs/dp);
    different lengths → different num_steps → FSDP grad-allreduce desync → hang.
    Or with dynamic batching, mismatched tensor shapes in the allreduce → hang.

    Also sets dynamic_global_batch_size so num_steps == 1 regardless of original gbs.
    """
    n = torch.tensor([len(rollout_data["tokens"])], device=torch.cuda.current_device())
    dist.all_reduce(n, op=dist.ReduceOp.MIN, group=dp_group)
    n_min = n.item()
    if micro_batch_size is not None:
        n_min = (n_min // micro_batch_size) * micro_batch_size
    assert n_min > 0, (
        f"cross-rank min(len(kept)) rounded to {n_min} — at least one rank has "
        f"no valid <explanation> extractions. Actor is not emitting tags "
        f"reliably. Raise rollout_batch_size or check actor SFT checkpoint."
    )
    out = {k: v[:n_min] for k, v in rollout_data.items()}
    out["dynamic_global_batch_size"] = n_min * dist.get_world_size(dp_group)
    return out


def _repartition_for_critic(rollout_data_ref, actor_dp, critic_rank, critic_dp):
    """Repartition actor_dp-split rollout data for a critic with different dp.

    Critic rank i takes actor partitions [i, i+critic_dp, i+2*critic_dp, ...].
    Fetches those from Ray, concatenates list-typed keys, re-wraps as a
    critic_dp-length list so process_rollout_data (data.py:273) sees the right
    len. All critic ranks share the same aggregated total_lengths (it's the
    full dataset's) but each has its own partition indices.
    """
    assert len(rollout_data_ref) == actor_dp, (
        f"expected {actor_dp} actor partitions, got {len(rollout_data_ref)}"
    )
    my_actor_parts = list(range(critic_rank, actor_dp, critic_dp))
    fetched = [ray.get(rollout_data_ref[i].inner) for i in my_actor_parts]

    # Concatenate: partition indices union, list-typed data keys append.
    # total_lengths is the FULL dataset's (same in all partitions, keep first).
    merged = {"total_lengths": fetched[0]["total_lengths"], "partition": []}
    for d in fetched:
        merged["partition"].extend(d["partition"])
        for k, v in d.items():
            if k in ("partition", "total_lengths"):
                continue
            if isinstance(v, list):
                merged.setdefault(k, []).extend(v)
            else:
                merged.setdefault(k, v)  # scalars/None: take first

    # Re-wrap: critic_dp Boxes. process_rollout_data does refs[dp_rank].inner,
    # so only our rank's Box needs real data. Others are None-inner placeholders
    # (never accessed). Box class: miles.utils.ray_utils.Box.
    new_refs = [Box(None)] * critic_dp
    new_refs[critic_rank] = Box(ray.put(merged))
    return new_refs


class NLATextOnlyCausalLM:
    """Auto-class shim: load + unwrap multimodal → text-only CausalLM.

    miles' FSDPTrainRayActor.get_model_cls() returns AutoModelForImageTextToText
    when hf_config has vision_config (Gemma-3 triggers this) → actor has
    vision_tower params → RL weight-sync to text-only sglang 400s on the first
    vision_tower key. resolve_text_model unwraps to a CausalLM wrapper around
    the text side only. No-op for Qwen/Llama/Mistral (no .language_model attr).

    The shim interface is the minimum miles needs: `.from_pretrained(...)` is
    the only callsite (fsdp_utils/actor.py: model_cls.from_pretrained(...)).
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return resolve_text_model(model)


class _SGLangKeyRemap:
    """Wrap a model so state_dict() keys match sglang's multimodal naming.

    Our actor (via NLATextOnlyCausalLM) is Gemma3ForCausalLM — keys `model.*`.
    sglang loads `google/gemma-3-12b-it` as Gemma3ForConditionalGeneration
    (architectures=['Gemma3ForConditionalGeneration'] in HF config) — keys
    `language_model.model.*`. Weight-sync iterates actor state_dict and sends
    names verbatim; sglang's load_weights does params_dict[name] → KeyError →
    HTTP 400 on the very first param.

    This wrapper prepends the prefix for weight-sync only (doesn't touch
    FSDP/training — weight_updater captures its own model ref at init).
    """

    def __init__(self, model: torch.nn.Module, prefix: str):
        self._model = model
        self._prefix = prefix

    def state_dict(self):
        return {self._prefix + k: v for k, v in self._model.state_dict().items()}


class NLAFSDPActor(FSDPTrainRayActor):

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        _nla_maybe_patch_fp32()  # NLA_FP32=1 -> fp32 compute (runs here so it lands in the worker)
        if role == "critic":
            assert args.critic_save is not None, (
                "NLA RL requires --critic-save (reward fn reads from there)"
            )
            args.hf_checkpoint = args.critic_load
            if args.critic_load_dcp:
                tracker = Path(args.critic_load_dcp) / "latest_checkpointed_iteration.txt"
                assert tracker.is_file(), (
                    f"--critic-load-dcp={args.critic_load_dcp!r} has no tracker file. "
                    f"checkpoint.load() would silently return None → critic keeps "
                    f"HF weights from --critic-load, ignoring the DCP overlay you asked for."
                )
            args.load = args.critic_load_dcp or args.critic_load
            args.save = args.critic_save
            args.lr = args.critic_lr or args.lr
            # Megatron wires this at megatron_utils/actor.py:93; FSDP doesn't.
            if args.critic_lr_warmup_iters:
                args.lr_warmup_iters = args.critic_lr_warmup_iters
            args.loss_type = "custom_loss"
            args.custom_loss_function_path = "nla.loss.nla_critic_loss"
            args.nla_model_is_critic = True
            # Critic's sidecar lives at critic_load — it has critic_num_layers,
            # mse_scale, suffix_ids. --nla-sidecar-source on CLI is the ACTOR's
            # override (for injection_scale from its model sidecar). Swap to the
            # critic's source: --nla-critic-sidecar-source if set, else None →
            # resolve_sidecar_source falls through to hf_checkpoint = critic_load.
            # (Megatron REQUIRES nla_critic_sidecar_source since its critic_load
            # is torch_dist with no sidecar; FSDP's fall-through to critic_load
            # HF dir is why None works here.)
            args.nla_sidecar_source = args.nla_critic_sidecar_source

        self._is_critic_model = getattr(args, "nla_model_is_critic", False)

        rollout_id = super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        # Parent keeps the full wrapper config (needs .vision_config for its own
        # checks); NLA only cares about text-side hidden_size/num_hidden_layers.
        self._text_config = resolve_text_config(self.hf_config)

        # sglang loads the HF checkpoint's architecture (multimodal for Gemma),
        # weight-sync sends OUR text-only keys. If we unwrapped, bridge with a
        # prefix remapper. weight_updater only exists for the actor role (sglang
        # sync path) — critic role doesn't create one. _text_config != hf_config
        # is the exact signal: we stripped a multimodal wrapper.
        if (
            not self._is_critic_model
            and self._text_config is not self.hf_config
            and hasattr(self, "weight_updater")
        ):
            prefix = resolve_sglang_key_prefix(self.hf_config)
            if prefix:
                self.weight_updater.model = _SGLangKeyRemap(self.model, prefix)

        assert get_parallel_state().cp.size == 1, (
            "NLA requires cp_size=1. With cp>1, slice_with_cp splits each sample "
            "into non-contiguous chunks; injection token + neighbors can land on "
            "different CP ranks, breaking the in-hook scan."
        )

        # get_grpo_returns (ppo_utils.py) takes kl but only uses it for .ones_like
        # (shape) — the value is discarded. So --kl-coef with grpo/gspo computes
        # ref_log_probs (slow!), builds the kl tensor, then throws it away. The
        # actual GRPO KL path is --use-kl-loss, which adds KL to the policy loss
        # instead (logs as train/kl_loss). This silently ate early RL runs.
        if role == "actor" and args.advantage_estimator in ("grpo", "gspo"):
            assert args.kl_coef == 0, (
                f"--kl-coef={args.kl_coef} is a NO-OP under "
                f"--advantage-estimator={args.advantage_estimator}: "
                f"get_grpo_returns discards the kl tensor. Use --use-kl-loss "
                f"--kl-loss-coef {args.kl_coef} instead (adds KL to policy loss, "
                f"logs as train/kl_loss). Or set --kl-coef 0 explicitly if you "
                f"don't want KL."
            )

        # Train/sample consistency guard. All released NLA runs train with
        # exactly ONE optimizer step per rollout (global batch == rollout
        # samples) — we have not tested anything else. A --global-batch-size
        # that doesn't equal rollout_batch_size × n_samples_per_prompt does NOT
        # give you more/fewer steps here (_train_core forces one step via the
        # dynamic_global_batch_size override), but Miles' loss wrapper still
        # normalizes by global_batch_size, so a mismatch silently rescales
        # gradients instead. Refuse to start rather than do either silently.
        if role == "actor" and args.loss_type == "policy_loss":
            rollout_samples = args.rollout_batch_size * args.n_samples_per_prompt
            if rollout_samples != args.global_batch_size:
                assert os.environ.get("NLA_I_KNOW_WHAT_IM_DOING") == "1", (
                    f"--rollout-batch-size {args.rollout_batch_size} × "
                    f"--n-samples-per-prompt {args.n_samples_per_prompt} = "
                    f"{rollout_samples} samples per rollout, but "
                    f"--global-batch-size is {args.global_batch_size}. NLA training "
                    f"has only been tested with one optimizer step per rollout "
                    f"(global batch == rollout samples). This mismatch would not "
                    f"actually change the step count — the FSDP path forces one "
                    f"step per rollout — but the loss is normalized by "
                    f"global_batch_size, so it silently scales gradients by "
                    f"{rollout_samples / args.global_batch_size:g}× instead. Make "
                    f"the two equal, or set NLA_I_KNOW_WHAT_IM_DOING=1 if you are "
                    f"deliberately running an untested configuration."
                )

        if role == "critic" and args.force_use_critic:
            actor_dp = args.actor_num_nodes * args.actor_num_gpus_per_node
            critic_dp = args.critic_num_nodes * args.critic_num_gpus_per_node
            # RolloutManager partitions by actor_dp (set by actor rank 0 via
            # miles/ray/train_actor.py set_train_parallel_config). Critic's
            # process_rollout_data asserts len(refs) == dp_size.
            # When dp differs, we stash actor_dp and repartition in train().
            assert critic_dp <= actor_dp, (
                f"critic_dp={critic_dp} > actor_dp={actor_dp} is not supported: "
                f"_repartition_for_critic distributes actor partitions across critic "
                f"ranks, so critic ranks >= actor_dp would get nothing. Reduce "
                f"CRITIC_NODES/CRITIC_GPUS so critic_dp <= actor_dp."
            )
            self._nla_actor_dp = actor_dp if actor_dp != critic_dp else None
            if self._nla_actor_dp is not None:
                print(f"[NLA] asymmetric DP: actor={actor_dp} critic={critic_dp}. "
                      f"Critic will fetch all {actor_dp} actor partitions and re-slice.")

        cfg, sidecar_source = load_nla_config_from_args(args, self.tokenizer)
        assert cfg.d_model == self._text_config.hidden_size, (
            f"sidecar d_model={cfg.d_model} != model hidden_size="
            f"{self._text_config.hidden_size}. Wrong checkpoint for this dataset."
        )
        if self._is_critic_model:
            # arguments.py:1796 defaults critic_load=load, so a missing
            # --critic-load silently loads the full-depth actor checkpoint.
            # Positive arch check catches that.
            assert cfg.critic_num_layers is not None, (
                f"critic model loaded from {args.hf_checkpoint!r} but sidecar "
                f"has no critic_num_layers. Did --critic-load default to the "
                f"actor checkpoint? Point it at the prepared K+1-layer critic."
            )
            assert self._text_config.num_hidden_layers == cfg.critic_num_layers + 1, (
                f"critic checkpoint has {self._text_config.num_hidden_layers} "
                f"layers, sidecar says extraction layer_index K="
                f"{cfg.critic_num_layers} → expect K+1="
                f"{cfg.critic_num_layers + 1} layers. Wrong checkpoint."
            )

        # injection_scale is a TRAINING HYPERPARAMETER — REQUIRED for actor
        # training. load_nla_config_from_args already applied any CLI override;
        # here we assert the value was resolved (via CLI, model sidecar, or
        # --nla-sidecar-source). Dataset sidecars deliberately don't carry
        # injection_scale — pick explicitly.
        #
        # INFERENCE MUST MATCH: nla_generate.py also calls load_nla_config_from_args
        # (same helper, same resolution), so train/infer scale cannot diverge.
        injects = not self._is_critic_model and args.loss_type in ("sft_loss", "policy_loss")
        if injects:
            assert cfg.injection_scale is not None, (
                "Actor training requires injection_scale. Set --nla-injection-scale "
                "(e.g. '150', 'raw', 'sqrt_d_model'), or point --nla-sidecar-source "
                "at a model sidecar that has it. Dataset sidecars don't carry it — "
                "it's a training hyperparameter, pick explicitly. "
                f"(Resolved sidecar: {sidecar_source!r}, injection_scale: None.)"
            )
        self._nla_cfg: NLAConfig = cfg
        self._nla_vectors: torch.Tensor | None = None
        # NLA forces one optimizer step per rollout by writing
        # rollout_data["dynamic_global_batch_size"] in _train_core (all paths).
        # Upstream asserts args.use_dynamic_global_batch_size == key-presence,
        # so flip the flag on THIS worker only (RolloutManager keeps the CLI
        # value and must NOT add the key — we compute it post-filter/truncate).
        self.args.use_dynamic_global_batch_size = True
        # Expose mse_scale on args so nla_critic_loss can read it backend-agnostically.
        # (Megatron's forward_step closure can't mutate batch; args is the shared channel.)
        self.args.nla_mse_scale = cfg.mse_scale

        # Predict-the-mean baselines for FVE. If passed via CLI (--nla-baseline-*,
        # precomputed from schema.compute_predict_mean_baselines), use those
        # directly — skips the init-time parquet read. Otherwise rank 0 reads
        # + broadcasts. Megatron uses CLI-only (no fallback compute).
        if self._is_critic_model and args.prompt_data is not None and args.nla_baseline_rawvar is None:
            baselines = [0.0]
            if dist.get_rank() == 0:
                t0 = time.perf_counter()
                source = args.prompt_data.split("@[")[0]
                if is_remote(source):
                    assert args.nla_storage_cls is not None
                    source = _load_storage(args.nla_storage_cls).open_read(source)
                _, b_rv = load_predict_mean_baselines(source, cfg.mse_scale)
                baselines[0] = b_rv
                dt = time.perf_counter() - t0
                print(f"[NLA] FVE baseline rawvar={b_rv:.4f} "
                      f"(mse_scale={cfg.mse_scale}, took {dt:.1f}s)")
            dist.broadcast_object_list(baselines, src=0)
            self.args.nla_baseline_rawvar = baselines[0]

        # miles calls gradient_checkpointing_enable() with no kwargs
        # (fsdp_utils/actor.py:123) — HF defaults to use_reentrant=True.
        # Reentrant checkpoint's backward re-runs forward via a custom
        # autograd.Function whose recompute does NOT trigger FSDP2's
        # post-forward reshard hook. All-gather buffers from the recompute
        # stay alive through the rest of backward. At 62 layers × 826MB
        # (27b) = 51GB pileup. 74GB OOM at rollout 1 once adam state lands.
        # Memory snapshot at the OOM: 54 × 826MB foreach_all_gather,
        # post-forward only 17GB (FWDMEM hook) → backward-only pileup.
        # Standalone FSDP test WITHOUT grad-ckpt → 10.64GB → confirms.
        #
        # use_reentrant=False (non-reentrant) runs recompute via a normal
        # forward call, module hooks fire, FSDP reshards correctly. PyTorch
        # FSDP docs explicitly recommend this. miles should fix upstream.
        # NLACriticModel.gradient_checkpointing_enable/_disable delegate
        # to backbone so this works for both roles.
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Hook registration AFTER grad-ckpt re-enable — defensive ordering.
        # HF's gradient_checkpointing_disable() clears forward hooks on
        # submodules (it removes checkpoint wrappers by clearing hooks), so
        # registering before the disable/re-enable cycle risks losing them.
        # Earlier 12b configs had no grad-ckpt re-enable so ordering was moot.
        if not self._is_critic_model and args.loss_type in ("sft_loss", "policy_loss"):
            self._register_injection_hook(self.model)
            if self.ref_model is not None:
                self._register_injection_hook(self.ref_model)

        return rollout_id

    def get_model_cls(self):
        if self._is_critic_model:
            return NLACriticModel
        # NLA is text-only. Parent returns AutoModelForImageTextToText when
        # hf_config has vision_config (Gemma-3) → actor has vision_tower params
        # → RL weight-sync to text-only sglang 400s. resolve_text_model unwraps
        # to Gemma3ForCausalLM (no-op for Qwen/Llama). See arch_adapters.py.
        return NLATextOnlyCausalLM

    def connect_actor_critic(self, critic_group):
        # Miles' PPO critic creates an actor↔critic NCCL group for syncing
        # per-token values into the actor's GAE computation (megatron_utils/actor.py:552).
        # NLA's critic is independent — GRPO advantages come from group-normed
        # rewards, not critic values. Both groups consume the same rollout_data_ref;
        # nothing to sync.
        pass

    def update_weights(self, info):
        """Sync actor weights to SGLang, then dump embedding for nla_generate.

        The rollout worker's cached embedding goes stale after each train step.
        Since the trainer is idle during rollout, and update_weights fires right
        before rollout starts, this is the moment to dump a fresh copy.
        nla_generate._maybe_reload_embed reads it.
        """
        super().update_weights(info)
        # debug_train_only (SFT mode): no SGLang rollout worker, so nla_generate
        # never runs → no consumer for the dump. Skip — saves ~2.2s/step
        # (FSDP all-gather of 1.1GB embedding + torch.save to disk).
        if (self._is_critic_model or self.args.save is None
                or self.args.debug_rollout_only or self.args.debug_train_only):
            return
        # --offload-train moves model to CPU before this (train.py:92 → sleep()
        # → model.cpu()). Mirror the parent updater's .cuda() (update_weight_utils.py:58)
        # so .full_tensor() runs its NCCL all-gather on GPU.
        weight = self.model.get_input_embeddings().weight.detach().cuda()
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        if dist.get_rank() == 0:
            out_path = embed_dump_path(self.args.save)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            torch.save(weight.detach().cpu(), tmp)
            tmp.rename(out_path)  # atomic — no mid-write read
        dist.barrier()

    def _register_injection_hook(self, model):
        embed = model.get_input_embeddings()
        inj = self._nla_cfg.injection_token_id
        left = self._nla_cfg.injection_left_neighbor_id
        right = self._nla_cfg.injection_right_neighbor_id

        def hook(_module, inputs, output):
            if self._nla_vectors is None or os.environ.get("NLA_SKIP_INJECTION") == "1":
                return output
            assert len(inputs) == 1 and inputs[0].dtype == torch.long
            return inject_at_marked_positions(
                input_ids=inputs[0],
                embeddings=output,
                vectors=self._nla_vectors,
                inj_id=inj, left_id=left, right_id=right,
            )

        embed.register_forward_hook(hook)

    def _get_model_inputs_args(self, batch):
        mm = batch.get("multimodal_train_inputs")
        if mm is not None and MM_ACTIVATION_KEY in mm:
            popped = mm.pop(MM_ACTIVATION_KEY)  # [B, d_model], raw from dataset
            if self._is_critic_model:
                batch[MM_ACTIVATION_KEY] = popped
                batch[MM_MSE_SCALE_KEY] = self._nla_cfg.mse_scale
            else:
                self._nla_vectors = normalize_activation(popped, self._nla_cfg.injection_scale)
        model_args = super()._get_model_inputs_args(batch)
        # use_cache=False kills TWO bugs, both via the same DynamicCache:
        #
        # (1) ref_lp mismatch: Gemma3TextModel.forward:518 creates DynamicCache when
        #     `use_cache and past_key_values is None and not self.training`.
        #     ref.eval() → cache; actor (train mode) → none. Gemma3 sliding-window
        #     attn picks different mask based on cache presence → ref_lp=-3.39 vs
        #     actor_lp=-1.32, identical weights. Qwen has no sliding window.
        #
        # (2) thd-packed cross-sequence contamination (miles passes
        #     attention_mask=None at fsdp_utils/actor.py:645). transformers HAS
        #     packed detection — masking_utils.py:735 infers block-diag from
        #     position_id resets — but it's gated on `past_key_values is None`.
        #     DynamicCache → detection bypassed → SDPA/eager fall through to full
        #     causal mask over the pack → seq N attends to seq 1..N-1.
        #     Measured: eval+default=4.2%
        #     L2 drift, eval+use_cache=False=0.6% (→0.0% in fp32, bf16 GEMM-tiling
        #     noise from batch-shape diff). Qwen FA2: varlen from position_ids
        #     directly, never touches this path.
        #
        # DO NOT remove — without this, every thd microbatch has contaminated
        # gradients on seqs 2..N. All training callers funnel through here
        # (_train_step, _compute_log_prob, _ref_log_probs_no_swap).
        model_args["use_cache"] = False
        return model_args

    def _create_ref_model(self, ref_load_path):
        # --nla-ref-on-gpu: UNTESTED since the hook + DynamicCache fixes (both
        # landed after this was dropped for a kl≈7 symptom that
        # turned out to be those bugs). Gives back the ~20s/step CPU swap at
        # ~7.5GB VRAM. Re-validate KL init ≈0 before using in production.
        if not getattr(self.args, "nla_ref_on_gpu", False):
            return super()._create_ref_model(ref_load_path)
        # Miles hardcodes cpu_offload=True for ref → ~20s/step actor↔ref CPU swap.
        # At m16+resp150: actor ~31GB + ref ~7.5GB (FSDP-sharded) = ~38GB, fits on 80GB.
        print(f"[NLA] --nla-ref-on-gpu: ref from {ref_load_path} stays on GPU "
              f"(skips ~20s/step swap, costs ~7.5GB VRAM)")
        with self._get_init_weight_context_manager()():
            ref = self.get_model_cls().from_pretrained(
                ref_load_path, trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
                # convert_fsdp_to_hf saves fp32 (DCP is fp32). Without this
                # cast, ref is 2× on GPU: 18GB sharded at DP=6 vs 9GB bf16.
                # 45GB pre-train baseline vs 36 expected → OOM at step 0.
                # (Same fix as actor.py:611 for the CPUOffload path.)
                torch_dtype=torch.bfloat16,
            )
        full_state = ref.state_dict()
        ref = apply_fsdp2(ref, mesh=get_parallel_state().dp_mesh, cpu_offload=False, args=self.args)
        ref = self._fsdp2_load_full_state_dict(ref, full_state, get_parallel_state().dp_mesh, cpu_offload=False)
        ref.cuda()  # from_pretrained→CPU, FSDP cpu_offload=False won't move it — pin to GPU now
        ref.eval()
        return ref

    def _compute_log_prob(self, model_tag, data_iterator, num_microbatches, store_prefix=""):
        # Critic model has no .logits. compute_advantages_and_returns early-returns
        # when log_probs/values are None (loss.py:315); get_batch returns None for
        # absent keys (data.py:300). Stock _train_core handles the rest.
        # sft_loss (loss.py:785-835) recomputes from logits — this pass is wasted
        # (full model forward, injection hook, clone — ~2× step time).
        if self._is_critic_model or self.args.loss_type == "sft_loss":
            return {}
        if (model_tag == "ref" and self.ref_model is not None
                and getattr(self.args, "nla_ref_on_gpu", False)):
            return self._ref_log_probs_no_swap(data_iterator, num_microbatches, store_prefix)
        return super()._compute_log_prob(model_tag, data_iterator, num_microbatches, store_prefix)

    def _ref_log_probs_no_swap(self, data_iterator, num_microbatches, store_prefix):
        # Same forward loop as parent's _compute_log_prob ref branch, but without
        # the model.cpu()/model.cuda() swap (ref is on-GPU already). Parent's version
        # is at fsdp_utils/actor.py:310-392 — this is that minus lines 318-321, 386-392.
        forward_data_store = []
        data_iterator.reset()
        with timer(f"{store_prefix}log_probs"), torch.no_grad():
            for step_id in range(len(num_microbatches)):
                for _ in self.prof.iterate_train_log_probs(
                    tqdm(range(num_microbatches[step_id]),
                         desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                ):
                    batch = get_batch(
                        data_iterator,
                        ["tokens", "loss_masks", "multimodal_train_inputs",
                         "total_lengths", "response_lengths", "max_seq_lens"],
                        self.args.data_pad_size_multiplier,
                        self.args.qkv_format,
                        get_position_ids=True,
                    )
                    model_args = self._get_model_inputs_args(batch)
                    logits = self.ref_model(**model_args).logits.float()
                    result = get_log_probs_and_entropy(
                        logits=logits, args=self.args,
                        unconcat_tokens=batch["unconcat_tokens"],
                        total_lengths=batch["total_lengths"],
                        response_lengths=batch["response_lengths"],
                        with_entropy=False,
                        max_seq_lens=batch.get("max_seq_lens", None),
                    )
                    forward_data_store.append({f"{store_prefix}log_probs": result["log_probs"]})
        return aggregate_forward_results(forward_data_store, data_iterator, self.args, store_prefix)

    def _train_step(self, batch, step_id, num_microbatches):
        if self._is_critic_model:
            model_args = self._get_model_inputs_args(batch)
            out = self.model(**model_args)
            # thd: [1, T, d] -> [T, d]; bshd: keep [B, L, d] (loss handles both)
            bb = out.backbone_last_hidden.detach()
            batch["_nla_backbone_last_hidden"] = bb.squeeze(0) if batch.get("max_seq_lens") is None else bb
            values = out.values.float()
            loss, _, log_dict = loss_function(
                args=self.args, batch=batch, num_microbatches=num_microbatches,
                logits=values,
            )
            loss.backward()
            return log_dict
        log_dict = super()._train_step(batch, step_id, num_microbatches)
        # FSDP2 overlaps this microbatch's reduce-scatter with the next
        # microbatch's all-gather prefetch. Gemma's 1.41B tied embedding
        # (262k vocab × 5376 d, root FSDP group per fsdp_utils/actor.py:687)
        # lands 5 embedding-sized tensors alive at the boundary = 16.9GB peak
        # on top of model + optimizer. Step 0 survives (no Adam state); once
        # Adam exists (+20GB), any seq-len variance tips it. Diagnosed via
        # torch.cuda.memory._dump_snapshot — _fsdp_collectives.py:508
        # foreach_reduce (5.64GB fp32) + :262 foreach_all_gather (2×2.82GB).
        # Synchronize forces reduce-scatter completion before next all-gather;
        # 3-tensor peak ~11GB. Loses some comm-compute overlap (~5-10% tput).
        # Different mechanism from use_reentrant (that breaks reshard during
        # recompute; this is the boundary between microbatches).
        torch.cuda.synchronize()
        return log_dict

    def critic_fwd(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Inference-only forward, returns values at each sample's last real token.

        Ray-callable from RolloutManager during generation (when trainer is idle).
        FSDP collective — ALL ranks must call this (RayTrainGroup.critic_fwd dispatches
        to every rank). Each rank computes identical output; caller takes rank 0's.

        Returns [B, d] CPU tensor (small → cheap to ship back over Ray object store).
        """
        assert self._is_critic_model, "critic_fwd called on non-critic actor"
        ids = input_ids.cuda(non_blocking=True)
        mask = attention_mask.cuda(non_blocking=True)
        # Rightmost True in mask, robust to either padding side.
        # GemmaTokenizerFast defaults to padding_side='left' (mask
        # [0,0,1,1,1]) where the old mask.sum-1 gave n_real-1 instead
        # of T-1. Qwen defaults right, so it worked by accident. In an
        # early Gemma RL run this picked the wrong pos for 31/32 samples —
        # actor chased an artificial length gradient (longer → less
        # padding → less-wrong idx) instead of explanation quality.
        last_idx = mask.cumsum(dim=1).argmax(dim=1)
        # no_grad, NOT inference_mode: the latter marks FSDP's gathered param
        # tensors as inference-only → next training forward crashes at F.linear
        # with "Inference tensors cannot be saved for backward". no_grad is
        # autograd-compatible.
        with torch.no_grad():
            values = self.model(input_ids=ids, attention_mask=mask, use_cache=False).values  # [B, T, d]
            out = values[torch.arange(ids.shape[0], device=ids.device), last_idx]  # [B, d]
        return out.float().cpu()

    def train(self, rollout_id, rollout_data_ref):
        # Asymmetric DP: RolloutManager partitions by actor_dp (set by actor
        # rank 0 via miles/ray/train_actor.py set_train_parallel_config).
        # When critic_dp < actor_dp, process_rollout_data asserts
        # len(refs) == dp_size → fails.
        # Repartition here: critic rank i takes actor partitions i, i+critic_dp,
        # i+2*critic_dp, ... . Fetch+concat+re-wrap. Uneven load if actor_dp %
        # critic_dp != 0 (e.g. 6→4: ranks 0,1 get 2 partitions, 2,3 get 1), but
        # all data is processed. No-op when _nla_actor_dp is None (equal DP,
        # Qwen's working path unchanged).
        if getattr(self, "_nla_actor_dp", None) is not None:
            rollout_data_ref = _repartition_for_critic(
                rollout_data_ref, self._nla_actor_dp,
                get_parallel_state().intra_dp.rank, get_parallel_state().intra_dp.size,
            )
        return super().train(rollout_id, rollout_data_ref)

    def _train_core(self, rollout_id, rollout_data):
        # All data prep happens here — parent's train() already did
        # get_rollout_data + timers + perf logging.
        if self._is_critic_model and self.role == "critic":
            if rollout_id == 0:
                # All ranks run (FSDP forward is collective); each sees its own
                # slice of rollout_data but per-sample ratios must all be ~1.0.
                # Rank-0-gated assert would leave other ranks hanging in FSDP
                # allgather when rank 0 dies — let the exception fire everywhere.
                _assert_reward_train_paths_agree(
                    self.critic_fwd, self.model, rollout_data, self._nla_cfg.mse_scale,
                    qkv_format=self.args.qkv_format,
                )
            rollout_data = _swap_rollout_to_critic_tokens(
                rollout_data, torch.cuda.current_device()
            )
            rollout_data = _truncate_to_cross_rank_min(
                rollout_data,
                get_parallel_state().intra_dp.group,
                None if self.args.use_dynamic_batch_size else self.args.micro_batch_size,
            )
            if self.args.qkv_format == "bshd":
                # get_rollout_data computed max_seq_lens from ACTOR tokens;
                # after the swap the batch holds critic tokens — recompute
                # (mirrors data.py's pad-to-multiple rounding).
                pad_size = get_parallel_state().tp.size * self.args.data_pad_size_multiplier
                m = max(rollout_data["total_lengths"])
                m = (m + pad_size - 1) // pad_size * pad_size
                rollout_data["max_seq_lens"] = [m] * len(rollout_data["tokens"])
        elif not self._is_critic_model:
            # LM-actor: strip variable-length critic tokens (would flow to
            # model(**kwargs) as unknown kwarg after multimodal concat).
            for mm in rollout_data.get("multimodal_train_inputs") or []:
                if mm is not None:
                    for k in CRITIC_ONLY_MM_KEYS:
                        mm.pop(k, None)
            # _compute_log_prob truncates to microbatch boundary (n // micro_bsz * micro_bsz)
            # but rollout_data's per-sample lists stay at the original length. With
            # indivisible counts (e.g. 512 rollouts / 3 DP = 170.67 → 171/170/170, then
            # 171 // 8 = 21 batches = 168), downstream gets len(rewards)=171 vs
            # len(log_probs)=168 → IndexError. Qwen's batch sizes were divisible.
            #
            # Cross-rank sync: each rank's n may differ (171/170/170) → different
            # n_aligned (168/168/168 here, but not guaranteed). get_data_iterator
            # reads dynamic_global_batch_size to compute num_microbatches — if ranks
            # disagree, one gets [] (all_reduce MIN hangs or asserts fail). Same
            # pattern as the critic's _truncate_to_cross_rank_min above.
            n_local = torch.tensor(
                [len(rollout_data.get("tokens", []))],
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(n_local, op=dist.ReduceOp.MIN, group=get_parallel_state().intra_dp.group)
            micro = self.args.micro_batch_size
            n_aligned = (n_local.item() // micro) * micro
            assert n_aligned > 0, (
                f"actor has {n_local.item()} samples after cross-rank MIN, "
                f"fewer than micro_batch_size={micro}. Raise rollout_batch_size."
            )
            n_orig = len(rollout_data.get("tokens", []))
            for k, v in list(rollout_data.items()):
                if isinstance(v, list) and len(v) == n_orig:
                    rollout_data[k] = v[:n_aligned]
            rollout_data["dynamic_global_batch_size"] = (
                n_aligned * dist.get_world_size(get_parallel_state().intra_dp.group)
            )
        if "dynamic_global_batch_size" not in rollout_data:
            # critic-SL standalone (_is_critic_model, role=="actor"): exact
            # global count via SUM — partitions may be uneven.
            n_local = torch.tensor(
                [len(rollout_data.get("tokens", []))],
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(n_local, op=dist.ReduceOp.SUM, group=get_parallel_state().intra_dp.group)
            rollout_data["dynamic_global_batch_size"] = int(n_local.item())
        super()._train_core(rollout_id=rollout_id, rollout_data=rollout_data)
        self._nla_vectors = None

    def save_model(self, rollout_id, force_sync=False):
        super().save_model(rollout_id, force_sync)
        if self.args.debug_rollout_only or self.args.save is None:
            return

        # get_model_state_dict with full_state_dict=True is a COLLECTIVE.
        # All ranks must call it or rank 0 deadlocks in the all-gather.
        # actor.py:96 doesn't pass torch_dtype → model stored fp32;
        # MixedPrecision is compute-only. Cast here for 2× smaller saves.
        full_sd = None
        if self._is_critic_model:
            full_sd = get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            full_sd = {
                k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v)
                for k, v in full_sd.items()
            }

        # Match fsdp_utils/checkpoint.py:199's iter_{rollout_id+1} convention.
        iter_dir = f"{self.args.save}/iter_{rollout_id + 1:07d}"

        if dist.get_rank() == 0:
            if self._is_critic_model:
                hf_dir = f"{iter_dir}/hf"
                self.model.save_pretrained(hf_dir, state_dict=full_sd)
                self.tokenizer.save_pretrained(hf_dir)
                # Write sidecar at BOTH hf/ and iter_N/ — the hf/ one is what
                # --critic-load needs (alongside config.json), the iter_N/ one
                # is a footgun defuser (if someone points at the wrong level).
                self._write_sidecar(hf_dir, rollout_id)
                self._write_sidecar(iter_dir, rollout_id)
            else:
                # Actor saves DCP only (model/, optimizer/). To load for RL:
                #   --hf-checkpoint = base model (from_pretrained needs safetensors)
                #   --load = this iter_dir (DCP overwrites weights)
                #   --nla-sidecar-source = this iter_dir (injection_scale from sidecar)
                # SGLang also loads base model; update_weights syncs SFT weights via NCCL.
                self._write_sidecar(iter_dir, rollout_id)
            keep_n = max(2, int(os.environ.get("NLA_KEEP_LOCAL", "2")))
            save_dir = self.args.save

            def _bg():
                self._maybe_background_push()
                if os.environ.get("NLA_BACKUP_REMOTE"):
                    prune = (f"ls -1d {save_dir}/iter_* 2>/dev/null | "
                             f"head -n -{keep_n} | xargs -r rm -rf")
                    subprocess.run(["bash", "-c", prune], check=False)

            threading.Thread(target=_bg, daemon=True).start()
        dist.barrier()

    def _maybe_background_push(self):
        """Fire-and-forget GCS push after each checkpoint, if NLA_BACKUP_REMOTE is set.

        For gs:// remotes, uses gsutil -m cp -r directly (no extra dep).
        For other schemes, goes through push_checkpoint + storage_cls.
        start_new_session detaches — upload survives later pkill of trainer.
        """
        remote = os.environ.get("NLA_BACKUP_REMOTE")
        if not remote:
            return
        # Serialize with the previous push: save_model's _bg thread prunes old
        # iter_* right after this returns. With keep_n=2 and one-behind push,
        # save N+1's prune deletes iter_{N-1} — which save N's gsutil may still
        # be uploading → truncated/corrupt remote. Block until that prior upload
        # finishes; the next prune is then safe.
        prev = getattr(self, "_push_proc", None)
        if prev is not None:
            prev.wait()
        role = "critic" if self._is_critic_model else "actor"
        remote_dir = f"{remote}/{role}"
        log = f"/tmp/push_{role}_iter.log"
        tracker = Path(self.args.save) / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            # First save with --async-save: super() started the write, no prior
            # to finalize → no tracker yet. Push is one-behind by design; nothing
            # to push on the first save.
            return
        latest = tracker.read_text().strip()
        iter_dir = f"{self.args.save}/iter_{int(latest):07d}"
        if remote.startswith("gs://"):
            if role == "actor":
                train_log = os.environ.get("NLA_TRAIN_LOG")
                if train_log and Path(train_log).exists():
                    shutil.copy(train_log, f"{iter_dir}/train.log")
                # RolloutManager writes sample_offset/epoch_id to a SIBLING
                # rollout/ dir — not inside iter_dir, so gsutil cp -r iter_dir
                # misses it. Snapshot into iter_dir so resume from GCS restores
                # data offset (else a fresh machine re-trains the first ~128k rows).
                rollout_state_dir = Path(self.args.save) / "rollout"
                if rollout_state_dir.exists():
                    for f in rollout_state_dir.glob("global_dataset_state_dict_*.pt"):
                        shutil.copy(f, f"{iter_dir}/{f.name}")
            # env -u PYTHONPATH: training's PYTHONPATH (Megatron-LM checkout)
            # can leak into gsutil's subprocess and break boto's Python-version
            # parsing under some Python distributions — strip it defensively.
            # Push only — caller handles prune (both backends: daemon thread
            # in save_model, push-then-prune). Chaining prune here previously
            # meant the `ls` readdir hung under async-save bg-write saturation.
            cmd = ["bash", "-c",
                   f"env -u PYTHONPATH gsutil -m cp -r {iter_dir} {remote_dir}/"]
        else:
            storage_cls = os.environ.get("NLA_BACKUP_STORAGE_CLS")
            assert storage_cls, "NLA_BACKUP_STORAGE_CLS required for non-gs:// remote"
            cmd = [sys.executable, "-m", "nla.scripts.push_checkpoint",
                   "--local", self.args.save, "--remote", remote_dir,
                   "--storage-cls", storage_cls, "--only-latest"]
        self._push_proc = subprocess.Popen(
            cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True
        )
        print(f"[NLA] background push fired: {iter_dir} → {remote_dir} (log: {log})")

    def _write_sidecar(self, checkpoint_dir: str, rollout_id: int):
        cfg = self._nla_cfg
        if self._is_critic_model:
            # num_hidden_layers is K+1 (blocks 0..K inclusive — we need
            # the output OF block K). Sidecar stores K (the extraction layer_index,
            # matching datagen's extraction.layer_index convention).
            cfg = replace(cfg, critic_num_layers=self._text_config.num_hidden_layers - 1)
        write_model_sidecar(
            checkpoint_dir, cfg,
            role="critic" if self._is_critic_model else "actor",
            stage="rl" if self.args.loss_type == "policy_loss" else "sl",
            base_checkpoint=self.args.hf_checkpoint,
            trained_on=[self.args.prompt_data] if self.args.prompt_data else [],
            parent_checkpoints=[self.args.hf_checkpoint],
            created_by="nla.train_actor.NLAFSDPActor",
            training_args={
                "rollout_id": rollout_id,
                "lr": self.args.lr,
                "loss_type": self.args.loss_type,
                "global_batch_size": self.args.global_batch_size,
            },
        )
