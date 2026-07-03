#!/bin/bash
# =============================================================================
# Evaluate an NLA RL (or SFT) checkpoint end to end:
#   base model -> last-token activations -> AV explanations -> AR cosine.
# Writes a parquet of {text, explanation, cosine_similarity}.
#
# Works on ANY checkpoint (not just the final one): point it at the actor/critic
# iter dirs. Base model and extraction layer are derived from the checkpoint
# (sidecar `base_checkpoint`; layer K = critic num_hidden_layers - 1), so it's
# not tied to Qwen-L20.
#
# Examples:
#   # an RL run laid out as <rl-dir>/{actor,critic}/iter_XXXXXXX
#   bash evaluate.sh --rl-dir /workspace/runs/rl --iter 400 \
#        --input-parquet /workspace/data/ultrafineweb_truncated.parquet \
#        --out /workspace/runs/eval_iter400.parquet
#
#   # explicit checkpoint dirs
#   bash evaluate.sh --actor-ckpt .../actor/iter_0000780 \
#        --critic-ckpt .../critic/iter_0000780 \
#        --input-parquet docs.parquet --out eval.parquet
#
#   # reuse cached activations across checkpoints (same base model + layer + docs)
#   bash evaluate.sh --rl-dir .../rl --iter 200 --acts-parquet .../acts.parquet \
#        --input-parquet docs.parquet --out eval_200.parquet
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
[ -n "${VIRTUAL_ENV:-}" ] || source /venv/main/bin/activate 2>/dev/null || true

# ---- defaults ----
RL_DIR=""; ITER=""; ACTOR_CKPT=""; CRITIC_CKPT=""
INPUT_PARQUET=""; OUT=""; TEXT_COLUMN="content"
TEMPERATURE="0.7"; MAX_NEW_TOKENS="200"; WORKERS="8"
WORK_DIR=""; ACTS_PARQUET=""; BASE_MODEL=""
SGLANG_PORT="30000"; AV_GPU="0"; AR_GPU="1"; N_ROWS=""; KEEP_RAW=""

while [ $# -gt 0 ]; do
  case "$1" in
    --rl-dir) RL_DIR="$2"; shift 2;;
    --iter) ITER="$2"; shift 2;;
    --actor-ckpt) ACTOR_CKPT="$2"; shift 2;;
    --critic-ckpt) CRITIC_CKPT="$2"; shift 2;;
    --input-parquet) INPUT_PARQUET="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --text-column) TEXT_COLUMN="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --work-dir) WORK_DIR="$2"; shift 2;;
    --acts-parquet) ACTS_PARQUET="$2"; shift 2;;
    --base-model) BASE_MODEL="$2"; shift 2;;
    --sglang-port) SGLANG_PORT="$2"; shift 2;;
    --av-gpu) AV_GPU="$2"; shift 2;;
    --ar-gpu) AR_GPU="$2"; shift 2;;
    --n-rows) N_ROWS="$2"; shift 2;;
    --keep-raw) KEEP_RAW="--keep-raw"; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# ---- resolve checkpoint dirs ----
if [ -n "$RL_DIR" ] && [ -n "$ITER" ]; then
  PAD="$(printf 'iter_%07d' "$ITER")"
  ACTOR_CKPT="${ACTOR_CKPT:-$RL_DIR/actor/$PAD}"
  CRITIC_CKPT="${CRITIC_CKPT:-$RL_DIR/critic/$PAD}"
fi
: "${ACTOR_CKPT:?set --actor-ckpt (or --rl-dir + --iter)}"
: "${CRITIC_CKPT:?set --critic-ckpt (or --rl-dir + --iter)}"
: "${INPUT_PARQUET:?set --input-parquet}"
: "${OUT:?set --out}"
[ -d "$ACTOR_CKPT/model" ]  || { echo "no DCP at $ACTOR_CKPT/model" >&2; exit 1; }
[ -d "$CRITIC_CKPT/model" ] || { echo "no DCP at $CRITIC_CKPT/model" >&2; exit 1; }

WORK_DIR="${WORK_DIR:-$(dirname "$OUT")/eval_work_$(basename "$ACTOR_CKPT")}"
AV_HF="$WORK_DIR/av_hf"; AR_HF="$WORK_DIR/ar_hf"
mkdir -p "$WORK_DIR"
echo "[evaluate] actor=$ACTOR_CKPT"
echo "[evaluate] critic=$CRITIC_CKPT"
echo "[evaluate] work_dir=$WORK_DIR  out=$OUT"

TOOLS="$REPO/tools"
# base model: from the actor sidecar unless overridden
RESOLVED_BM="${BASE_MODEL:-$(python -c "import yaml;print(yaml.safe_load(open('$ACTOR_CKPT/nla_meta.yaml'))['base_checkpoint'])")}"
echo "[evaluate] base_model=$RESOLVED_BM"

# ---- 1. AV (actor) DCP -> HF: the generic converter handles the plain CausalLM;
#         we just add the tokenizer + sidecar that SGLang/NLAClient also need.
python "$TOOLS/convert_fsdp_to_hf.py" --input-dir "$ACTOR_CKPT" --output-dir "$AV_HF" --origin-hf-dir "$RESOLVED_BM"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$RESOLVED_BM', trust_remote_code=True).save_pretrained('$AV_HF')"
cp "$ACTOR_CKPT/nla_meta.yaml" "$AV_HF/nla_meta.yaml"

# ---- 2. AR (critic) DCP -> HF: NLACriticModel (backbone + value_head) — the
#         generic converter can't represent it, so use the value-head-aware one.
python "$TOOLS/convert_critic_fsdp_to_hf.py" --critic-dcp "$CRITIC_CKPT" --out "$AR_HF" --base-model "$RESOLVED_BM"

# ---- 3. activations (skip if a precomputed parquet was supplied) ----
LAYER="$(python -c "import json;print(json.load(open('$AR_HF/config.json'))['num_hidden_layers']-1)")"
echo "[evaluate] extraction layer K=$LAYER"
if [ -z "$ACTS_PARQUET" ]; then
  ACTS_PARQUET="$WORK_DIR/acts.parquet"
  NR_ARG=(); [ -n "$N_ROWS" ] && NR_ARG=(--n-rows "$N_ROWS")
  python "$HERE/extract_activations.py" --base-model "$RESOLVED_BM" --layer "$LAYER" \
    --input-parquet "$INPUT_PARQUET" --text-column "$TEXT_COLUMN" \
    --out "$ACTS_PARQUET" "${NR_ARG[@]}"
else
  echo "[evaluate] reusing activations: $ACTS_PARQUET"
fi

# ---- 4. serve AV via SGLang (--disable-radix-cache REQUIRED for input_embeds) ----
SGLOG="$WORK_DIR/sglang.log"
CUDA_VISIBLE_DEVICES="$AV_GPU" python -m sglang.launch_server \
  --model-path "$AV_HF" --port "$SGLANG_PORT" \
  --disable-radix-cache --mem-fraction-static 0.85 \
  --context-length 512 --trust-remote-code >"$SGLOG" 2>&1 &
SGLANG_PID=$!
cleanup() { kill "$SGLANG_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[evaluate] waiting for SGLang (pid $SGLANG_PID) on port $SGLANG_PORT ..."
for _ in $(seq 1 180); do
  curl -sf "http://localhost:$SGLANG_PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SGLANG_PID" 2>/dev/null || { echo "SGLang died; see $SGLOG" >&2; tail -20 "$SGLOG" >&2; exit 1; }
  sleep 2
done
curl -sf "http://localhost:$SGLANG_PORT/health" >/dev/null 2>&1 || { echo "SGLang not ready" >&2; exit 1; }

# ---- 5. AV generate + AR score -> output parquet ----
python "$HERE/run_eval.py" \
  --av-hf "$AV_HF" --ar-hf "$AR_HF" --acts-parquet "$ACTS_PARQUET" \
  --out "$OUT" --text-column "$TEXT_COLUMN" \
  --sglang-url "http://localhost:$SGLANG_PORT" \
  --temperature "$TEMPERATURE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --workers "$WORKERS" --ar-device "cuda:$AR_GPU" $KEEP_RAW

echo "[evaluate] DONE -> $OUT"
