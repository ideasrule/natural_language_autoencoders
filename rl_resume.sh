#!/bin/bash
# =============================================================================
# Resume the Qwen3.6-27B NLA RL run from /workspace/old_runs/rl (rollout 160).
#
# Run with NO arguments and NO edits:   bash rl_resume.sh
#
# This is configs/rl.sh's train.py invocation inlined (per request, rl.sh
# itself is untouched) with the flags changed from "initialize RL from SFT"
# to "continue a previous RL run":
#
#   * --load points at the OLD RUN's actor save root (tracker -> iter 160)
#     and --finetune is NOT passed, so miles restores weights + optimizer +
#     lr-scheduler + rollout_id (160) + the dataset position
#     ({load}/rollout/global_dataset_state_dict_159.pt). Training continues
#     at rollout 160 of the ~3905-rollout epoch.
#   * --critic-load is the old run's critic HF export (architecture/config +
#     value_head); --critic-load-dcp overlays its DCP (weights + optimizer,
#     tracker-resolved to iter 160).
#   * --ref-load (KL reference, coef 0.01) is the old run's actor-SFT HF
#     export, staged at /workspace/old_runs/av_sft/iter_0000970/hf.
#   * Saves go back into the SAME directories (in-place resume, per request).
#     A background pruner keeps the 2 newest iters per role — NOTE this will
#     eventually delete the original iter_0000160 (accepted trade-off).
#
# Precision: actor resumes bf16; the critic runs fp32-storage + tf32-matmul
# (NLA_FP32_CRITIC + NLA_TF32_CRITIC). A bf16-critic resume was tried first
# (the old run's DCP is bf16 and trained 160 clean rollouts on ITS box): the
# forward restored perfectly (loss 0.162, fve 0.641) but the very first
# backward on THIS box produced grad_norm=nan — the same bf16 fragility the
# from-scratch smoke hit, so it is hardware/stack-dependent, not just an
# undertrained-critic artifact. Consequences of the fp32 switch:
#   * the critic loads its weights from the iter-160 HF export (identical
#     tensors to the DCP; from_pretrained upcasts bf16->fp32 cleanly) and
#     --critic-load-dcp is NOT passed — the bf16 DCP optimizer state can't
#     load into fp32 Adam, so the critic restarts with fresh momenta
#     (supervised MSE; re-adapts within tens of steps).
#   * the critic worker flips itself to sdpa (FA rejects fp32); tf32 keeps
#     its step ~70s (validated in the smoke: grad norms match plain fp32).
#
# Box specifics carried over from train_qwen3.6.sh (see its header):
#   * 8x B200: actor=4 / critic=2 / rollout=2 (critic_dp must divide actor_dp)
#   * NCCL_PROTO=Simple + NCCL_NVLS_ENABLE=0 (LL-protocol small-allreduce
#     deadlock on this stack, flight-recorder verified)
#   * --no-save-optim + save_model's empty-dir cleanup (disk budget; NEW saves
#     are weights-only — a crash resumes weights-only from the last save)
#   * soft length penalty 0.0025/token past 140; response cap 200, ctx 384
# =============================================================================
set -euo pipefail

MILES=/workspace/miles
VENV=/venv/main
OLD_RL=/workspace/old_runs/rl
ACTOR_ITER="$OLD_RL/actor/iter_0000160"
# NOT iter_0000160/hf: that export's value_head.safetensors is corrupt
# (25215 NaN + 218 Inf of 26.2M elements — scattered, likely a bad copy;
# symptom: finite backbone_norm but NaN pred_norm/rewards from the first
# forward). critic_hf_clean is rebuilt from the iter-160 DCP (the clean
# source of truth) via tools/convert_critic_fsdp_to_hf.py.
CRITIC_ITER_HF="$OLD_RL/critic_hf_clean"
REF_HF_CKPT=/workspace/old_runs/av_sft/iter_0000970/hf
RL_PARQUET=/workspace/data/rl_shuf.parquet
BASE_MODEL=Qwen/Qwen3.6-27B
NO_THINK='{"enable_thinking": false}'

export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export NLA_EMBED_DUMP_DIR=/dev/shm/nla
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_PROTO=Simple
export NCCL_NVLS_ENABLE=0
export NLA_FP32_CRITIC=1
export NLA_TF32_CRITIC=1
export NLA_LEN_PENALTY_START=140
export NLA_LEN_PENALTY_SLOPE=0.0025
mkdir -p "$NLA_EMBED_DUMP_DIR"

source "$VENV/bin/activate"

# --- preflight: every checkpoint piece the resume depends on ----------------
for f in "$OLD_RL/actor/latest_checkpointed_iteration.txt" \
         "$ACTOR_ITER/nla_meta.yaml" \
         "$ACTOR_ITER/model/.metadata" \
         "$CRITIC_ITER_HF/config.json" \
         "$CRITIC_ITER_HF/value_head.safetensors" \
         "$REF_HF_CKPT/config.json" \
         "$RL_PARQUET.nla_meta.yaml"; do
    [ -e "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
# The corrupt-export incident above: refuse to launch on non-finite weights.
python - <<'PY'
from safetensors.torch import load_file
import torch, sys
w = load_file("/workspace/old_runs/rl/critic_hf_clean/value_head.safetensors")["weight"]
bad = torch.isnan(w).sum().item() + torch.isinf(w).sum().item()
sys.exit(f"FATAL: value_head has {bad} non-finite elements" if bad else 0)
PY

# --- background pruner: keep the 2 newest iters per role (newest may be
# mid-write; the previous one is the last complete save) ----------------------
prune_rl_iters() {
    while true; do
        for role in actor critic; do
            ls -d "$OLD_RL/$role"/iter_* 2>/dev/null | sort -V | head -n -2 | xargs -r rm -rf
        done
        sleep 300
    done
}
prune_rl_iters & PRUNER_PID=$!
trap 'kill "$PRUNER_PID" 2>/dev/null || true' EXIT

cd "$MILES"
python train.py \
    --train-backend fsdp \
    --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
    --loss-type policy_loss \
    --advantage-estimator grpo \
    --force-use-critic \
    --n-samples-per-prompt 8 \
    --rollout-function-path miles.rollout.sglang_rollout.generate_rollout \
    --custom-generate-function-path nla.rollout.nla_generate.generate \
    --custom-rm-path nla.reward.nla_rm \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data "$RL_PARQUET" \
    --input-key prompt \
    --hf-checkpoint "$BASE_MODEL" \
    --ref-load "$REF_HF_CKPT" \
    --use-kl-loss --kl-loss-coef 0.01 \
    `# RESUME (vs rl.sh): --load is the old RL actor root, NO --finetune ->` \
    `# optimizer/lr-scheduler/rollout_id/dataset position all restore.` \
    --load "$OLD_RL/actor" \
    --nla-sidecar-source "$ACTOR_ITER" \
    --save "$OLD_RL/actor" \
    `# NO --critic-load-dcp: fp32 critic can't ingest the bf16 DCP optimizer;` \
    `# weights come from the HF export below (same iter-160 tensors).` \
    --critic-load "$CRITIC_ITER_HF" \
    --critic-save "$OLD_RL/critic" \
    --critic-lr 1.41e-5 \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --critic-num-nodes 1 \
    --critic-num-gpus-per-node 2 \
    --rollout-num-gpus 2 \
    --rollout-max-response-len 200 \
    --rollout-max-context-len 384 \
    `# REQUIRED for NLA — radix cache keys on token IDs, but we inject raw` \
    `# activation vectors at the marker token. DO NOT REMOVE to "optimize".` \
    --sglang-disable-radix-cache \
    --sglang-context-length 384 \
    --router-history-backend none \
    --router-policy round_robin \
    --router-disable-circuit-breaker \
    --router-retry-max-backoff-ms 500 --router-retry-max-retries 2 \
    --rollout-batch-size 128 \
    --global-batch-size 1024 \
    --micro-batch-size 4 \
    --lr 1.41e-5 --lr-decay-style constant \
    --num-epoch 1 \
    --save-interval 200 \
    --qkv-format bshd \
    --no-save-optim \
    --loss-mask-type qwen \
    --apply-chat-template-kwargs "$NO_THINK"

echo "=== NLA Qwen3.6-27B RL resume complete: actor=$OLD_RL/actor critic=$OLD_RL/critic ==="
