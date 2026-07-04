#!/bin/bash
# =============================================================================
# Train the Qwen3.6-27B NLA end to end on this 8xB200-183GB box.
#
#   stage 0  prepare critic init checkpoint (truncate Qwen3.6 to layers 0..42)
#   stage 1  AR / critic SFT      (MSE on raw activations)
#   stage 2  AV / actor  SFT      (next-token on explanations, with injection)
#   stage 3  convert actor DCP -> HF   (needed for the RL KL reference model)
#   stage 4  RL                   (GRPO AV + supervised AR, reward = -mse_nrm)
#
# Run with NO arguments and NO edits:   bash train_qwen3.6.sh
#
# Assumptions baked in (this exact box / data):
#   * 8x B200-183GB.
#   * Data at $DATA (the *_shuf.parquet splits + .nla_meta.yaml sidecars,
#     regenerated for Qwen/Qwen3.6-27B layer 42, d_model 5120).
#   * Miles checkout at /workspace/miles (NLA patches applied), patched SGLang
#     at /workspace/sglang, venv at /venv/main, HF cache at /workspace/.hf_home.
#     (setup_qwen3.6.sh produces all of this.)
#
# -----------------------------------------------------------------------------
# Qwen3.6-27B specifics (vs the Qwen3.5-9B run this is adapted from):
#   * SAME architecture class (Qwen3_5ForConditionalGeneration, model_type
#     qwen3_5): hybrid 3:1 gated-deltanet:full attention, vision wrapper the
#     code strips, SAME 248320-token tokenizer and injection contract
#     (158983, neighbors 29/510 — verified by setup_qwen3.6.sh). Every
#     Qwen3.5 code path (bshd batching, sglang qwen3_5 input_embeds bypass,
#     empty-''-key weight-sync prefix, thinking-disabled template split)
#     applies unchanged.
#   * Layer 42 of 64, d_model 5120 (from the dataset sidecar).
#   * INJ_SCALE=90 — median raw activation L2 norm at layer 42 measured from
#     the av_sft training parquet (median 90.8, p10-p90 78.5-102.5; sqrt_d
#     would be 71.6). Same "ambient residual scale" rationale as Qwen3.5's 75.
#   * THINKING DISABLED everywhere, same mechanism as Qwen3.5 (see that
#     script's header): --apply-chat-template-kwargs '{"enable_thinking":
#     false}' in both SFT and RL. sft_actor.py asserts the head/full prefix
#     split per sample, so template drift fails loudly.
#   * Precision: same recipe — NLA_FP32=1 + sdpa for the SFT stages; bf16 RL.
#   * 8-GPU sizing: SFT stages use all 8 GPUs (grad accum from batch 256).
#     Actor micro-batch 4 + gradient checkpointing (27B fp32: sharded
#     param+grad+opt is ~54GB/GPU before activations). Critic micro-batch 8
#     (43-layer/~19B backbone). RL layout actor=4 / critic=2 / rollout=2:
#     critic_dp MUST evenly divide actor_dp (_repartition_for_critic strides
#     partitions round-robin; guarded by an assert in nla/train_actor.py).
#     Under NLA_FP32_CRITIC the critic's fp32 param+grad+Adam shard is
#     ~152GB/GPU at dp=2 — tight on 183GB, so expandable_segments caps
#     fragmentation. The bf16 actor at dp=4 sits ~85GB/GPU incl. ref model.
#   * DISK (~1TB budget on /workspace): --no-save-optim on EVERY stage.
#     Optimizer states are never consumed downstream (RL loads the SFT actor
#     with --finetune = weights-only) and cost ~2x model size per save
#     (27B fp32 optim alone is ~216GB). A crash therefore restarts the
#     affected stage from its last weights-only save (fresh optimizer) —
#     acceptable for ~1000-step SFT stages and constant-LR RL. Stage 4 also
#     runs a background pruner that keeps only the 2 newest RL iters per role.
# =============================================================================
set -euo pipefail

NLA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILES=/workspace/miles
VENV=/venv/main
DATA=/workspace/data
RUNS=/workspace/runs
BASE_MODEL=Qwen/Qwen3.6-27B
LAYER=42
INJ_SCALE=90
NO_THINK='{"enable_thinking": false}'
export HF_HOME=/workspace/.hf_home
# Everything (base model, tokenizer) is pre-cached by setup_qwen3.6.sh; offline
# mode stops from_pretrained's per-rank hub revalidation, which intermittently
# 404s under concurrent load ("does not appear to have a file named ...").
export HF_HUB_OFFLINE=1

mkdir -p "$RUNS"
source "$VENV/bin/activate"

# --- guard: the fp32 monkeypatch must be present in NLA (see header) -----------
grep -q "_nla_maybe_patch_fp32" "$NLA_REPO/nla/train_actor.py" || {
    echo "FATAL: fp32 monkeypatch (_nla_maybe_patch_fp32) missing from" \
         "$NLA_REPO/nla/train_actor.py — the SFT stages would NaN in bf16." >&2
    exit 1
}

# =============================================================================
# Stage 0 — prepare the critic init checkpoint (K=42 -> keeps blocks 0..42).
# =============================================================================
cd "$NLA_REPO"
if [ ! -f "$RUNS/critic_init/nla_meta.yaml" ]; then
python nla/scripts/prepare_critic_checkpoint.py \
    --base-model "$BASE_MODEL" \
    --num-layers "$LAYER" \
    --dataset-sidecar "$DATA/ar_sft_shuf.parquet" \
    --output "$RUNS/critic_init"
fi

# =============================================================================
# Stage 1 — AR (critic) SFT.  fp32 + sdpa.  8 GPUs, micro-batch 8.
# Configs invoke train.py by relative path, so every stage runs from Miles.
# =============================================================================
cd "$MILES"
if [ ! -f "$RUNS/critic_sft/latest_checkpointed_iteration.txt" ]; then
NLA_FP32=1 \
AR_SFT_PARQUET="$DATA/ar_sft_shuf.parquet" \
CRITIC_INIT_CKPT="$RUNS/critic_init" \
SAVE_DIR="$RUNS/critic_sft" \
bash "$NLA_REPO/configs/critic_sft.sh" \
    --actor-num-gpus-per-node 8 \
    --micro-batch-size 8 \
    --attn-implementation sdpa \
    `# bshd (padded batches, NOT thd packing): the gated-deltanet layers` \
    `# ignore sequence boundaries in a packed stream (transformers hardcodes` \
    `# seq_idx=None) -> recurrent state leaks across samples. bshd isolates` \
    `# samples on the batch dim; right-padding is causal-safe.` \
    --qkv-format bshd \
    --no-save-optim
fi

# =============================================================================
# Stage 2 — AV (actor) SFT.  fp32 + sdpa + gradient checkpointing.
# injection_scale=90 (see header); thinking disabled via chat-template kwargs.
# =============================================================================
cd "$MILES"
if [ ! -f "$RUNS/actor_sft/latest_checkpointed_iteration.txt" ]; then
NLA_FP32=1 \
AV_SFT_PARQUET="$DATA/av_sft_shuf.parquet" \
INSTRUCT_MODEL="$BASE_MODEL" \
SAVE_DIR="$RUNS/actor_sft" \
INJ_SCALE="$INJ_SCALE" \
bash "$NLA_REPO/configs/actor_sft.sh" \
    --actor-num-gpus-per-node 8 \
    --micro-batch-size 4 \
    --attn-implementation sdpa \
    --gradient-checkpointing \
    --qkv-format bshd \
    --apply-chat-template-kwargs "$NO_THINK" \
    --no-save-optim
fi

# resolve the iter dirs the SFT stages just wrote (resolve dynamically so it
# reproduces regardless of exact step count).
ACTOR_ITER="$(ls -d "$RUNS"/actor_sft/iter_* | sort -V | tail -1)"
CRITIC_ITER_HF="$(ls -d "$RUNS"/critic_sft/iter_*/hf | sort -V | tail -1)"

# =============================================================================
# Stage 3 — convert the actor SFT DCP -> HF.  The RL KL reference model loads
# via from_pretrained (HF format); actor_sft.sh only writes a DCP.
# =============================================================================
cd "$NLA_REPO"
if [ ! -f "$RUNS/actor_sft_hf/config.json" ]; then
python tools/convert_fsdp_to_hf.py \
    --input-dir "$ACTOR_ITER" \
    --output-dir "$RUNS/actor_sft_hf" \
    --origin-hf-dir "$BASE_MODEL"
fi

# Disk: the critic init (38GB) is consumed only by stage 1, and intermediate
# (non-final) SFT iters are superseded by the final ones. Prune once the
# artifacts downstream stages need are confirmed present.
if [ -f "$RUNS/actor_sft_hf/config.json" ] && [ -d "$CRITIC_ITER_HF" ]; then
    rm -rf "$RUNS/critic_init"
    for d in "$RUNS"/actor_sft/iter_* ; do [ "$d" != "$ACTOR_ITER" ] && rm -rf "$d"; done
    for d in "$RUNS"/critic_sft/iter_* ; do [ "$d" != "$(dirname "$CRITIC_ITER_HF")" ] && rm -rf "$d"; done
fi

# =============================================================================
# Stage 4 — RL (GRPO AV + supervised AR), KL=0.01.  bf16.  8-GPU disjoint
# layout: actor=4 / critic=2 / rollout=2 (see header).  One epoch over the
# RL split.
# Thinking disabled in rollout prompts via the same chat-template kwargs.
# =============================================================================
# Soft length penalty (reward -= 0.0025 * max(0, len-140)): GRPO's
# within-group advantage keeps favoring longer explanations after the
# genuine quality gradient saturates, and the token-mean KL loss is
# length-blind — without this fence the policy ratchets to the cap and
# collapses (measured on Qwen3.5, same tokenizer + explanation corpus;
# carried over unchanged).
#
# Background checkpoint pruner: keep the 2 newest iter_* per role (the newest
# may be mid-write; the one before it is the last complete save). Weights-only
# saves (no optimizer) are ~54GB actor / ~38GB critic.
prune_rl_iters() {
    while true; do
        for role in actor critic; do
            ls -d "$RUNS/rl/$role"/iter_* 2>/dev/null | sort -V | head -n -2 | xargs -r rm -rf
        done
        sleep 300
    done
}
prune_rl_iters & PRUNER_PID=$!
trap 'kill "$PRUNER_PID" 2>/dev/null || true' EXIT

# NCCL_PROTO=Simple + NCCL_NVLS_ENABLE=0: NCCL LL/LL128 small-allreduce
# deadlock on this B200 stack — the critic's first grad-clip allreduce (574
# elems) hangs with matched op streams (flight-recorder verified). Only
# small-message latency is affected.
# NLA_FP32_CRITIC=1: critic bf16 backward NaNs on the 27B (finite forward,
# NaN grads at step 0, weights then poisoned through the unguarded clip) ->
# fp32 for the critic group only (gates the fp32 monkeypatch per-role; the
# critic worker flips itself to sdpa since FlashAttention rejects fp32). The
# GRPO actor stays bf16+FA2 — it trains clean and fp32 would blow its memory.
# NLA_TF32_CRITIC=1: recover the fp32 slowdown (287s -> 70s critic step) via
# tf32 matmuls; fp32 range/storage/activations keep the NaN fix (validated:
# step-0 grad_norm 2.2896 tf32 vs 2.2851 fp32; bf16 was NaN).
cd "$MILES"
NLA_EMBED_DUMP_DIR=/dev/shm/nla \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_PROTO=Simple \
NCCL_NVLS_ENABLE=0 \
NLA_FP32_CRITIC=1 \
NLA_TF32_CRITIC=1 \
ACTOR_NODES=1 ACTOR_GPUS=4 \
CRITIC_NODES=1 CRITIC_GPUS=2 \
ROLLOUT_GPUS=2 \
KL_LOSS_COEF=0.01 \
NLA_LEN_PENALTY_START=140 \
NLA_LEN_PENALTY_SLOPE=0.0025 \
ACTOR_SFT_CKPT="$ACTOR_ITER" \
REF_HF_CKPT="$RUNS/actor_sft_hf" \
CRITIC_SL_CKPT="$CRITIC_ITER_HF" \
INSTRUCT_MODEL="$BASE_MODEL" \
RL_PARQUET="$DATA/rl_shuf.parquet" \
RUN_DIR="$RUNS/rl" \
bash "$NLA_REPO/configs/rl.sh" \
    --num-epoch 1 \
    --save-interval 200 \
    --qkv-format bshd \
    `# empty optimizer/ dirs from --no-save-optim are rmdir'd by` \
    `# NLAFSDPActor.save_model so resume loads skip them cleanly` \
    --no-save-optim \
    `# Response cap 200 (rl.sh default 150): same tokenizer + explanation` \
    `# distribution as the Qwen3.5 run, where the critic reward kept improving` \
    `# with explanation length until ~140 tokens and a cap at 150 made GRPO` \
    `# chase the gradient into the wall (truncation -> FAILED(-2) -> sawtooth).` \
    `# 200 puts the wall past the saturation point. Context lens follow` \
    `# (prompt ~90 + response).` \
    --rollout-max-response-len 200 \
    --rollout-max-context-len 384 \
    --sglang-context-length 384 \
    --apply-chat-template-kwargs "$NO_THINK"

echo "=== NLA Qwen3.6-27B training complete: actor=$RUNS/rl/actor critic=$RUNS/rl/critic ==="
