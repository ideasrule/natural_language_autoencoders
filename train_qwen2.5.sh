#!/bin/bash
# =============================================================================
# Reproduce the Qwen2.5-7B NLA training run end to end on this 4xH100-80GB box.
#
#   stage 0  prepare critic init checkpoint (truncate Qwen to layers 0..20)
#   stage 1  AR / critic SFT      (MSE on raw activations)
#   stage 2  AV / actor  SFT      (next-token on explanations, with injection)
#   stage 3  convert actor DCP -> HF   (needed for the RL KL reference model)
#   stage 4  RL                   (GRPO AV + supervised AR, reward = -mse_nrm)
#
# Run with NO arguments and NO edits:   bash train_qwen2.5.sh
# It runs the four stages in the foreground, in order, each blocking until done.
#
# Assumptions baked in (this exact box / data):
#   * 4x H100-80GB.
#   * Training data already generated at $DATA (the *_shuf.parquet splits).
#   * Miles checkout at /workspace/miles, venv at /venv/main, HF cache populated.
#
# -----------------------------------------------------------------------------
# Precision: the SFT stages need fp32 — bf16 produced NaN grads (finite forward,
# non-finite backward) within the first few steps. Miles is NOT modified; fp32 is
# forced by an env-gated monkeypatch in nla/train_actor.py (_nla_maybe_patch_fp32,
# run from NLAFSDPActor.init so it lands in the Ray worker): NLA_FP32=1 forces both
# fp32 storage (the NLA model loaders' from_pretrained) and fp32 compute (FSDP
# MixedPrecisionPolicy). fp32 needs sdpa (FlashAttention rejects fp32). The guard
# below verifies the monkeypatch is present. configs/rl.sh carries the matching
# --load(root)+--finetune and REF_HF_CKPT fixes.
#
# RL runs bf16 (fp32 won't fit the actor on dp=2 without grad-ckpt, and grad-ckpt
# deadlocks RL update_weights()). It was numerically stable in bf16 here; there is
# no NaN-skip guard (that lived in a Miles edit we reverted — it's in the optimizer
# loop, not reachable by the monkeypatch trick).
# =============================================================================
set -euo pipefail

# --- paths / constants (this is the NLA repo dir, wherever the script lives) ---
NLA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILES=/workspace/miles
VENV=/venv/main
DATA=/workspace/data/nla_qwen7b_ultrafineweb_20k
RUNS=/workspace/runs
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
export HF_HOME=/workspace/.hf_home

mkdir -p "$RUNS"
source "$VENV/bin/activate"

# --- guard: the fp32 monkeypatch must be present in NLA (see header) -----------
grep -q "_nla_maybe_patch_fp32" "$NLA_REPO/nla/train_actor.py" || {
    echo "FATAL: fp32 monkeypatch (_nla_maybe_patch_fp32) missing from" \
         "$NLA_REPO/nla/train_actor.py — the SFT stages would NaN in bf16." >&2
    exit 1
}

# =============================================================================
# Stage 0 — prepare the critic init checkpoint (K=20 -> keeps blocks 0..20).
# =============================================================================
cd "$NLA_REPO"
python nla/scripts/prepare_critic_checkpoint.py \
    --base-model "$BASE_MODEL" \
    --num-layers 20 \
    --dataset-sidecar "$DATA/ar_sft_shuf.parquet" \
    --output "$RUNS/critic_init"

# =============================================================================
# Stage 1 — AR (critic) SFT.  fp32 + sdpa (fp32 needs sdpa; FlashAttn rejects
# fp32).  4 GPUs, micro-batch 8.  Configs invoke train.py by relative path, so
# every stage runs from the Miles checkout.
# =============================================================================
cd "$MILES"
NLA_FP32=1 \
AR_SFT_PARQUET="$DATA/ar_sft_shuf.parquet" \
CRITIC_INIT_CKPT="$RUNS/critic_init" \
SAVE_DIR="$RUNS/critic_sft" \
bash "$NLA_REPO/configs/critic_sft.sh" \
    --actor-num-gpus-per-node 4 \
    --micro-batch-size 8 \
    --attn-implementation sdpa

# =============================================================================
# Stage 2 — AV (actor) SFT.  fp32 + sdpa + gradient checkpointing (fp32 ~2x's
# activation memory on the full 28-layer 7B).  injection_scale=150, qwen mask.
# =============================================================================
cd "$MILES"
NLA_FP32=1 \
AV_SFT_PARQUET="$DATA/av_sft_shuf.parquet" \
INSTRUCT_MODEL="$BASE_MODEL" \
SAVE_DIR="$RUNS/actor_sft" \
INJ_SCALE=150 \
LOSS_MASK_TYPE=qwen \
bash "$NLA_REPO/configs/actor_sft.sh" \
    --actor-num-gpus-per-node 4 \
    --micro-batch-size 8 \
    --attn-implementation sdpa \
    --gradient-checkpointing

# resolve the iter dirs the SFT stages just wrote (epoch 1 -> iter_0000193 here,
# but resolve dynamically so it reproduces regardless of exact step count).
ACTOR_ITER="$(ls -d "$RUNS"/actor_sft/iter_* | sort -V | tail -1)"
CRITIC_ITER_HF="$(ls -d "$RUNS"/critic_sft/iter_*/hf | sort -V | tail -1)"

# =============================================================================
# Stage 3 — convert the actor SFT DCP -> HF.  The RL KL reference model loads
# via from_pretrained (HF format); actor_sft.sh only writes a DCP.
# =============================================================================
cd "$NLA_REPO"
python tools/convert_fsdp_to_hf.py \
    --input-dir "$ACTOR_ITER" \
    --output-dir "$RUNS/actor_sft_hf" \
    --origin-hf-dir "$BASE_MODEL"

# =============================================================================
# Stage 4 — RL (GRPO AV + supervised AR), KL=0.01.  bf16 (NLA_FP32 NOT set: fp32
# won't fit the actor on dp=2 without grad-ckpt, and grad-ckpt deadlocks RL
# update_weights()).  4-GPU disjoint layout: actor=2 / critic=1 / rollout=1.
# rl.sh derives the actor --load root from ACTOR_SFT_CKPT's parent and adds
# --finetune; REF_HF_CKPT is the HF export from stage 3; CRITIC_SL_CKPT is the
# critic SFT HF dir.  One epoch over the RL split; checkpoint every 50 rollouts.
# =============================================================================
cd "$MILES"
NLA_EMBED_DUMP_DIR=/dev/shm/nla \
ACTOR_NODES=1 ACTOR_GPUS=2 \
CRITIC_NODES=1 CRITIC_GPUS=1 \
ROLLOUT_GPUS=1 \
KL_LOSS_COEF=0.01 \
ACTOR_SFT_CKPT="$ACTOR_ITER" \
REF_HF_CKPT="$RUNS/actor_sft_hf" \
CRITIC_SL_CKPT="$CRITIC_ITER_HF" \
INSTRUCT_MODEL="$BASE_MODEL" \
RL_PARQUET="$DATA/rl_shuf.parquet" \
RUN_DIR="$RUNS/rl" \
bash "$NLA_REPO/configs/rl.sh" \
    --num-epoch 1 \
    --save-interval 50

echo "=== NLA Qwen2.5-7B training complete: actor=$RUNS/rl/actor critic=$RUNS/rl/critic ==="
