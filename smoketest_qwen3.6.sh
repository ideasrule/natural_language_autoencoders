#!/bin/bash
# =============================================================================
# Smoke test the Qwen3.6-27B NLA pipeline end to end at real batch/micro sizes
# but tiny step counts, into $RUNS_SMOKE (separate from the real $RUNS).
#
#   stage 0  critic init  — SHARED with the real run ($RUNS/critic_init):
#            deterministic truncation, no training state, 38GB; no point
#            writing it twice.
#   stage 1  AR / critic SFT   4 steps  (memory + loss-finite + hf export)
#   stage 2  AV / actor  SFT   4 steps  (injection + template prefix asserts)
#   stage 3  DCP -> HF convert (RL ref model plumbing)
#   stage 4  RL                2 rollouts (sglang injection, reward, weight sync)
#
# Everything mirrors train_qwen3.6.sh — same GPUs, micro batches, flags — so
# an OOM or a model-class incompatibility shows up here, in minutes not hours.
#
# Usage:  bash smoketest_qwen3.6.sh            # all stages
#         bash smoketest_qwen3.6.sh rl         # just stage 4 (reuses earlier)
# Cleanup after success:  rm -rf /workspace/runs_smoke
# =============================================================================
set -euo pipefail

NLA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILES=/workspace/miles
VENV=/venv/main
DATA=/workspace/data
RUNS=/workspace/runs            # critic_init shared with the real run
RUNS_SMOKE=/workspace/runs_smoke
BASE_MODEL=Qwen/Qwen3.6-27B
LAYER=42
INJ_SCALE=90
NO_THINK='{"enable_thinking": false}'
export HF_HOME=/workspace/.hf_home
# Everything (base model, tokenizer) is pre-cached by setup_qwen3.6.sh; offline
# mode stops from_pretrained's per-rank hub revalidation, which intermittently
# 404s under concurrent load ("does not appear to have a file named ...").
export HF_HUB_OFFLINE=1

ONLY="${1:-all}"
mkdir -p "$RUNS" "$RUNS_SMOKE"
source "$VENV/bin/activate"

run_stage() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

# ── stage 0: critic init (shared with real run) ─────────────────────────────
if [ ! -f "$RUNS/critic_init/nla_meta.yaml" ]; then
    cd "$NLA_REPO"
    python nla/scripts/prepare_critic_checkpoint.py \
        --base-model "$BASE_MODEL" \
        --num-layers "$LAYER" \
        --dataset-sidecar "$DATA/ar_sft_shuf.parquet" \
        --output "$RUNS/critic_init"
fi

# ── stage 1: critic SFT, 4 steps ────────────────────────────────────────────
if run_stage critic; then
    cd "$MILES"
    NLA_FP32=1 \
    AR_SFT_PARQUET="$DATA/ar_sft_shuf.parquet" \
    CRITIC_INIT_CKPT="$RUNS/critic_init" \
    SAVE_DIR="$RUNS_SMOKE/critic_sft" \
    bash "$NLA_REPO/configs/critic_sft.sh" \
        --actor-num-gpus-per-node 8 \
        --micro-batch-size 8 \
        --attn-implementation sdpa \
        --qkv-format bshd \
        --no-save-optim \
        --num-rollout 4 \
        --lr-warmup-iters 1 \
        --save-interval 4
fi

# ── stage 2: actor SFT, 4 steps ─────────────────────────────────────────────
if run_stage actor; then
    cd "$MILES"
    NLA_FP32=1 \
    AV_SFT_PARQUET="$DATA/av_sft_shuf.parquet" \
    INSTRUCT_MODEL="$BASE_MODEL" \
    SAVE_DIR="$RUNS_SMOKE/actor_sft" \
    INJ_SCALE="$INJ_SCALE" \
    bash "$NLA_REPO/configs/actor_sft.sh" \
        --actor-num-gpus-per-node 8 \
        --micro-batch-size 4 \
        --attn-implementation sdpa \
        --gradient-checkpointing \
        --qkv-format bshd \
        --apply-chat-template-kwargs "$NO_THINK" \
        --no-save-optim \
        --num-rollout 4 \
        --lr-warmup-iters 1 \
        --save-interval 4
fi

ACTOR_ITER="$(ls -d "$RUNS_SMOKE"/actor_sft/iter_* | sort -V | tail -1)"
CRITIC_ITER_HF="$(ls -d "$RUNS_SMOKE"/critic_sft/iter_*/hf | sort -V | tail -1)"

# ── stage 3: DCP -> HF (KL reference model) ─────────────────────────────────
if run_stage convert || run_stage rl; then
    if [ ! -f "$RUNS_SMOKE/actor_sft_hf/config.json" ]; then
        cd "$NLA_REPO"
        python tools/convert_fsdp_to_hf.py \
            --input-dir "$ACTOR_ITER" \
            --output-dir "$RUNS_SMOKE/actor_sft_hf" \
            --origin-hf-dir "$BASE_MODEL"
    fi
fi

# ── stage 4: RL, 2 rollouts ─────────────────────────────────────────────────
# NCCL_PROTO=Simple + NCCL_NVLS_ENABLE=0: NCCL LL/LL128 small-allreduce
# deadlock on this B200 stack — the critic's first grad-clip allreduce (574
# elems) hangs with matched op streams (flight-recorder verified). Only
# small-message latency is affected.
# NLA_FP32_CRITIC=1: critic bf16 backward NaNs on the 27B (finite forward,
# NaN grads at step 0) -> fp32 for the critic group only; its worker flips
# itself to sdpa.
if run_stage rl; then
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
    REF_HF_CKPT="$RUNS_SMOKE/actor_sft_hf" \
    CRITIC_SL_CKPT="$CRITIC_ITER_HF" \
    INSTRUCT_MODEL="$BASE_MODEL" \
    RL_PARQUET="$DATA/rl_shuf.parquet" \
    RUN_DIR="$RUNS_SMOKE/rl" \
    bash "$NLA_REPO/configs/rl.sh" \
        --num-rollout 2 \
        --save-interval 2 \
        --qkv-format bshd \
        `# empty optimizer/ dirs from --no-save-optim are rmdir'd by` \
        `# NLAFSDPActor.save_model so resume loads skip them cleanly` \
        --no-save-optim \
        --rollout-max-response-len 200 \
        --rollout-max-context-len 384 \
        --sglang-context-length 384 \
        --apply-chat-template-kwargs "$NO_THINK"
fi

echo "=== smoke test complete — artifacts in $RUNS_SMOKE (rm -rf when done) ==="
