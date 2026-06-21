#!/bin/bash
# NLA RL: simultaneous AV (GRPO) + AR (supervised MSE).
#
# The AR is trained alongside the AV because it IS the reward model: reward =
# -MSE(AR_fwd(explanation), gold_activation), computed via Ray remote on the
# live AR-trainer GPUs (nla/reward.py). A frozen AR would give stale rewards
# the AV would game — the AR must learn alongside the AV what semantic content
# each activation direction encodes.
#
# Two RayTrainGroups: actor (GRPO + injection) + critic (MSE). Both consume the
# same rollout_data_ref each step.
#
# Prerequisite: ACTOR_SFT_CKPT and CRITIC_SL_CKPT must both have nla_meta.yaml
# with matching token IDs / prompt templates.
#
# Defaults reproduce the released Qwen2.5-7B run: 128 prompts/rollout x 8
# samples/prompt = 1024-sample global batch, lr 1.41e-5, KL loss coef 0.01,
# 150-token response cap. See TRAINING_NOTES.md "Production run" for the full
# table and how these relate to the values recorded in the released
# checkpoints' nla_meta.yaml sidecars.
#
# Throughput knobs (micro-batch size, attention implementation, dynamic
# batching) were not profiled carefully and are not optimised for any
# particular hardware (including the H100s we ran on) — worth tuning for
# your setup. They change step time, not the training math.
#
# KEEP TRAINING ON-POLICY. This script uses Miles' synchronous train.py:
# generate -> train -> sync weights to SGLang, every step, one optimizer step
# per rollout (128 prompts x 8 samples = 1024 = global batch). That is the
# ONLY configuration we have tested — all released checkpoints were trained
# this way. Two specific cautions:
#   - One step per rollout: NLAFSDPActor refuses to start if
#     rollout_batch_size x n_samples_per_prompt != global_batch_size; set
#     NLA_I_KNOW_WHAT_IM_DOING=1 to bypass. (A mismatch would not change the
#     step count — the FSDP path forces one step — it silently rescales
#     gradients via the loss normalizer instead.)
#   - No overlapped generation/training: with Miles' train_async.py, rollout
#     N+1 is generated while rollout N trains, so samples come from slightly
#     stale weights. In our experience this train/sample mismatch can hurt
#     training, but we have not investigated it carefully. Stick with
#     train.py unless you know what you're doing.

: "${RL_PARQUET:?set RL_PARQUET to the Stage 3c parquet path}"
: "${INSTRUCT_MODEL:?HF base instruct model (e.g. Qwen/Qwen2.5-7B-Instruct) — supplies tokenizer/config}"
: "${ACTOR_SFT_CKPT:?actor_sft.sh iter dir (e.g. .../iter_0002000) — holds nla_meta.yaml for --nla-sidecar-source; --load uses its PARENT (the save root, which holds latest_checkpointed_iteration.txt). Pointing --load at the iter dir itself finds no tracker and SILENTLY skips the load -> actor stays base weights.}"
: "${CRITIC_SL_CKPT:?HF dir from critic_sft.sh (e.g. .../iter_0002000/hf) — already truncated, K is in its config.json}"
: "${RUN_DIR:?}"

# --kl-coef is a NO-OP for GRPO (get_grpo_returns discards the kl tensor).
# --use-kl-loss is the correct path (adds KL to policy loss, logs train/kl_loss)
# but it's action="store_true" — once passed, callers can't un-pass it. Gate on
# env var so KL_LOSS_COEF=0 drops the flags entirely (small-scale test runs use
# this to skip the reference model + its DCP→HF conversion).
#
# The KL reference model is the frozen SFT actor, created via from_pretrained, so
# --ref-load needs an HF checkpoint dir — NOT the DCP in ACTOR_SFT_CKPT. actor_sft.sh
# only writes a DCP, so convert it once (the "DCP→HF conversion step"):
#   python tools/convert_fsdp_to_hf.py --input-dir "$ACTOR_SFT_CKPT" \
#       --output-dir <REF_HF_CKPT> --origin-hf-dir "$INSTRUCT_MODEL"
# and pass that dir as REF_HF_CKPT. Required when KL_LOSS_COEF != 0; unused at 0.
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
# awk does the float compare natively; exit status = (k==0): 0 (shell-true) when
# nonzero -> KL on, 1 (shell-false) when zero -> KL off. Handles 0, 0.0, 1e-3.
if awk -v k="$KL_LOSS_COEF" 'BEGIN{exit (k==0)}'; then
    : "${REF_HF_CKPT:?KL_LOSS_COEF != 0 needs a reference model — set REF_HF_CKPT to an HF export of the actor SFT ckpt (tools/convert_fsdp_to_hf.py; see note above)}"
    KL_FLAGS=(--use-kl-loss --kl-loss-coef "$KL_LOSS_COEF")
    REF_FLAGS=(--ref-load "$REF_HF_CKPT")
else
    KL_FLAGS=()
    REF_FLAGS=()
fi

# Per-step 1.1GB embedding dump for nla_generate — /tmp is disk (overlay fs),
# /dev/shm is tmpfs (RAM). ~1.5s → ~0.1s per step. Zero code change.
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"
mkdir -p "$NLA_EMBED_DUMP_DIR"

# actor_dp may differ from critic_dp since _repartition_for_critic —
# critic rank i pulls actor partitions [i, i+critic_dp, ...]. Defaults keep
# them symmetric for standalone test runs.
ACTOR_NODES=${ACTOR_NODES:-1}
ACTOR_GPUS=${ACTOR_GPUS:-8}
CRITIC_NODES=${CRITIC_NODES:-$ACTOR_NODES}
CRITIC_GPUS=${CRITIC_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}

# Dynamic batching is OFF by default (--use-dynamic-batch-size not set).
# With micro-batch-size=4: critic gets 4 samples/microbatch regardless of length.
# To enable (packs by token budget instead — critic tokens are ~300 each):
#   --use-dynamic-batch-size --max-tokens-per-gpu 4096
# Safe for the critic: _swap_rollout_to_critic_tokens sets total_lengths to
# critic token lengths, which is what get_data_iterator reads for packing.

${PYTHON:-python} train.py \
    --train-backend "${TRAIN_BACKEND:-fsdp}" \
    --custom-actor-cls-path "${ACTOR_CLS:-nla.train_actor.NLAFSDPActor}" \
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
    --hf-checkpoint "$INSTRUCT_MODEL" \
    `# --ref-load (HF dir) present only when KL_LOSS_COEF != 0; see REF_HF_CKPT above` \
    "${REF_FLAGS[@]}" \
    `# --load must be the SAVE ROOT (parent of iter_XXXX) so Miles' tracker resolves` \
    `# the iter; the iter dir itself silently skips. --finetune = weights-only load` \
    `# (fresh rollout_id/lr; warm optimizer momenta).` \
    --load "$(dirname "${ACTOR_SFT_CKPT%/}")" \
    --finetune \
    --nla-sidecar-source "$ACTOR_SFT_CKPT" \
    --save "$RUN_DIR/actor" \
    --critic-load "$CRITIC_SL_CKPT" \
    --critic-save "$RUN_DIR/critic" \
    --critic-lr "${CRITIC_LR:-1.41e-5}" \
    --actor-num-nodes "$ACTOR_NODES" \
    --actor-num-gpus-per-node "$ACTOR_GPUS" \
    --critic-num-nodes "$CRITIC_NODES" \
    --critic-num-gpus-per-node "$CRITIC_GPUS" \
    --rollout-num-gpus "$ROLLOUT_GPUS" \
    --rollout-max-response-len 150 \
    --rollout-max-context-len 300 \
    `# REQUIRED for NLA — radix cache keys on token IDs, but we inject raw activation` \
    `# vectors at the marker token. Cache would hit across DIFFERENT activations that` \
    `# share the same marker token → silent wrong output. DO NOT REMOVE to "optimize".` \
    --sglang-disable-radix-cache \
    --sglang-context-length 300 \
    --router-history-backend none \
    `# cache_aware (default) builds prefix tree storing request bodies — with NLA` \
    `# input_embeds (~6-12MB each) that IS the leak. round_robin: no tree.` \
    `# CB-disable + short retry: large bodies drop connections, CB false-positive` \
    `# locks engines 60s. NLA workload is uniform — no routing smarts needed.` \
    --router-policy round_robin \
    --router-disable-circuit-breaker \
    --router-retry-max-backoff-ms 500 --router-retry-max-retries 2 \
    --rollout-batch-size 128 \
    --global-batch-size 1024 \
    --micro-batch-size "${ACTOR_MICRO:-4}" \
    --lr "${ACTOR_LR:-1.41e-5}" --lr-decay-style constant \
    "${KL_FLAGS[@]}" \
    --save-interval "${SAVE_INTERVAL:-100}" \
    --loss-mask-type "${LOSS_MASK_TYPE:-qwen}" \
    "$@"
