#!/bin/bash
# =============================================================================
# Train the Qwen3.5-9B NLA end to end on this 4xH200-141GB box.
#
#   stage 0  prepare critic init checkpoint (truncate Qwen3.5 to layers 0..21)
#   stage 1  AR / critic SFT      (MSE on raw activations)
#   stage 2  AV / actor  SFT      (next-token on explanations, with injection)
#   stage 3  convert actor DCP -> HF   (needed for the RL KL reference model)
#   stage 4  RL                   (GRPO AV + supervised AR, reward = -mse_nrm)
#
# Run with NO arguments and NO edits:   bash train_qwen3.5.sh
#
# Assumptions baked in (this exact box / data):
#   * 4x H200-141GB.
#   * Data at $DATA (the *_shuf.parquet splits + .nla_meta.yaml sidecars,
#     regenerated for Qwen/Qwen3.5-9B layer 21, d_model 4096).
#   * Miles checkout at /workspace/miles (NLA patches applied), patched SGLang
#     at /workspace/sglang, venv at /venv/main, HF cache at /workspace/.hf_home.
#
# -----------------------------------------------------------------------------
# Qwen3.5-9B specifics (vs the Qwen2.5-7B run this is adapted from):
#   * Layer 21 of 32, d_model 4096 (from the dataset sidecar).
#   * INJ_SCALE=75 — median raw activation L2 norm at layer 21 measured from
#     the training parquet (74-75; sqrt_d would be 64). Mirrors the "ambient
#     residual scale" rationale behind Qwen2.5's 150.
#   * THINKING DISABLED everywhere: --apply-chat-template-kwargs
#     '{"enable_thinking": false}' makes the chat template end the generation
#     prompt with the empty '<think>\n\n</think>\n\n' block. nla/rollout/
#     sft_actor.py splits the loss mask at exactly that boundary, and
#     nla_generate.py builds RL prompts with the same kwargs, so SFT-trained
#     positions == RL-generated positions, no thinking tokens anywhere.
#   * Hybrid attention (3:1 gated-deltanet:full): the linear-attention layers
#     use fla/causal-conv1d Triton kernels; --attn-implementation only routes
#     the full-attention layers.
#   * RL runs with KL 0.01 (user requirement; also used in the smoke test).
#   * Precision: same recipe as Qwen2.5 — NLA_FP32=1 (fp32 storage+compute via
#     the monkeypatch in nla/train_actor.py) + sdpa for the SFT stages; bf16
#     for RL. The GDN layers keep their internal fp32 state math either way.
# =============================================================================
set -euo pipefail

NLA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILES=/workspace/miles
VENV=/venv/main
DATA=/workspace/data
RUNS=/workspace/runs
BASE_MODEL=Qwen/Qwen3.5-9B
LAYER=21
INJ_SCALE=75
NO_THINK='{"enable_thinking": false}'
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
# Stage 0 — prepare the critic init checkpoint (K=21 -> keeps blocks 0..21).
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
# Stage 1 — AR (critic) SFT.  fp32 + sdpa.  4 GPUs, micro-batch 8.
# Configs invoke train.py by relative path, so every stage runs from Miles.
# =============================================================================
cd "$MILES"
if [ ! -f "$RUNS/critic_sft/latest_checkpointed_iteration.txt" ]; then
NLA_FP32=1 \
AR_SFT_PARQUET="$DATA/ar_sft_shuf.parquet" \
CRITIC_INIT_CKPT="$RUNS/critic_init" \
SAVE_DIR="$RUNS/critic_sft" \
bash "$NLA_REPO/configs/critic_sft.sh" \
    --actor-num-gpus-per-node 4 \
    --micro-batch-size 8 \
    --attn-implementation sdpa \
    `# bshd (padded batches, NOT thd packing): Qwen3.5's gated-deltanet layers` \
    `# ignore sequence boundaries in a packed stream (transformers hardcodes` \
    `# seq_idx=None) -> recurrent state leaks across samples. bshd isolates` \
    `# samples on the batch dim; right-padding is causal-safe.` \
    --qkv-format bshd
fi

# =============================================================================
# Stage 2 — AV (actor) SFT.  fp32 + sdpa + gradient checkpointing.
# injection_scale=75 (see header); thinking disabled via chat-template kwargs.
# =============================================================================
cd "$MILES"
if [ ! -f "$RUNS/actor_sft/latest_checkpointed_iteration.txt" ]; then
NLA_FP32=1 \
AV_SFT_PARQUET="$DATA/av_sft_shuf.parquet" \
INSTRUCT_MODEL="$BASE_MODEL" \
SAVE_DIR="$RUNS/actor_sft" \
INJ_SCALE="$INJ_SCALE" \
bash "$NLA_REPO/configs/actor_sft.sh" \
    --actor-num-gpus-per-node 4 \
    --micro-batch-size 8 \
    --attn-implementation sdpa \
    --gradient-checkpointing \
    --qkv-format bshd \
    --apply-chat-template-kwargs "$NO_THINK"
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

# =============================================================================
# Stage 4 — RL (GRPO AV + supervised AR), KL=0.01.  bf16.  4-GPU disjoint
# layout: actor=2 / critic=1 / rollout=1.  One epoch over the RL split.
# Thinking disabled in rollout prompts via the same chat-template kwargs.
# =============================================================================
# Soft length penalty (reward -= 0.0025 * max(0, len-140)): GRPO's
# within-group advantage keeps favoring longer explanations after the
# genuine quality gradient saturates (~140 tokens, measured), and the
# token-mean KL loss is length-blind — without this fence the policy
# ratchets to the cap and collapses (runs 1-3; kl-coef 5x did NOT help).
# With it (run 4): lengths hold ~125, best rewards of any run, no drift.
cd "$MILES"
NLA_EMBED_DUMP_DIR=/dev/shm/nla \
ACTOR_NODES=1 ACTOR_GPUS=2 \
CRITIC_NODES=1 CRITIC_GPUS=1 \
ROLLOUT_GPUS=1 \
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
    `# Response cap 200 (rl.sh default 150): under the Qwen3.5 tokenizer the` \
    `# critic reward keeps improving with explanation length until ~140 tokens` \
    `# (measured: mse 0.94@40 -> 0.30@100 -> 0.246@140, flat after), and the` \
    `# SFT init already puts 3.7% of responses >=150. With the cap at 150 GRPO` \
    `# chases the gradient into the wall: truncation -> FAILED(-2) -> sawtooth` \
    `# length/reward oscillations of growing amplitude (observed at rollouts` \
    `# 75-97). 200 puts the wall past the saturation point; the policy settles` \
    `# ~140 with ~0 truncations. Context lens follow (prompt ~90 + response).` \
    --rollout-max-response-len 200 \
    --rollout-max-context-len 384 \
    --sglang-context-length 384 \
    --apply-chat-template-kwargs "$NO_THINK"

echo "=== NLA Qwen3.5-9B training complete: actor=$RUNS/rl/actor critic=$RUNS/rl/critic ==="
