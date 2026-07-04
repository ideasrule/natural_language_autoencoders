#!/bin/bash
# =============================================================================
# Set up a fresh machine to train the Qwen3.6-27B NLA (companion to
# train_qwen3.6.sh — run this once, then that).
#
# Qwen3.6-27B is architecturally the same class as Qwen3.5-9B
# (Qwen3_5ForConditionalGeneration, model_type qwen3_5: hybrid 3:1
# gated-deltanet:full attention, vision wrapper, same 248320-token
# tokenizer, same injection-token contract 158983 with neighbors 29/510) —
# so this is setup_qwen3.5.sh with only the HF model swapped. Every patch
# (miles NLA integration, sglang qwen3_5 input_embeds bypass) applies
# unchanged.
#
# Produces:
#   * miles @ c8322c8 (radixark/miles) + the NLA integration patches
#   * sglang @ 6ee17b4 (sgl-project/sglang, `sglang-miles` branch) + the NLA
#     input_embeds/transport patches — installing sglang[all] brings the
#     pinned torch 2.11.0+cu130 / transformers 5.8.1 stack with it
#   * flash-linear-attention + causal-conv1d (built from source against the
#     local CUDA 13 toolkit — the gated-deltanet layers need both)
#   * this repo (nla) editable, Qwen/Qwen3.6-27B in the HF cache (~57GB)
#
# Assumptions (the Vast.ai PyTorch image satisfies all of these):
#   * NVIDIA driver + CUDA 13 toolkit at /usr/local/cuda (nvcc on PATH)
#   * a python venv at $VENV with `uv` available
#   * ~150GB free under $WORKSPACE (checkouts + HF model + wheels)
#   * root or passwordless sudo (one apt package: protobuf-compiler)
#
# Idempotent: safe to re-run; every step checks before doing work.
#
# Training data is NOT produced here — the *_shuf.parquet splits +
# .nla_meta.yaml sidecars (Qwen3.6-27B layer 42) must already be at $DATA.
# =============================================================================
set -euo pipefail

NLA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-/venv/main}"
DATA="${DATA:-$WORKSPACE/data}"
MILES_DIR="$WORKSPACE/miles"
SGLANG_DIR="$WORKSPACE/sglang"
BASE_MODEL="Qwen/Qwen3.6-27B"

# Pins this run was built against.
MILES_PIN="$(cut -d@ -f2 "$NLA_REPO/nla/miles_patches/UPSTREAM_PIN")"   # c8322c8
SGLANG_PIN="6ee17b43682436621125264eadccf3b9d4cc08ca"                    # sglang-miles branch

export HF_HOME="${HF_HOME:-$WORKSPACE/.hf_home}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

source "$VENV/bin/activate"
# uv resolves across the pytorch cu130 index + PyPI throughout.
UV_FLAGS=(--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130)

echo "=== [1/8] system packages (protoc for sglang's grpc build) ==="
if ! command -v protoc >/dev/null; then
    ${SUDO:-} apt-get update -q && ${SUDO:-} apt-get install -y -q protobuf-compiler
fi

echo "=== [2/8] rust toolchain (sglang v0.5.13+ builds a Rust component) ==="
if ! command -v rustc >/dev/null && [ ! -x "$HOME/.cargo/bin/rustc" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
fi
. "$HOME/.cargo/env" 2>/dev/null || true

echo "=== [3/8] sglang checkout @ $SGLANG_PIN + NLA patches ==="
if [ ! -d "$SGLANG_DIR/.git" ]; then
    git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
fi
git -C "$SGLANG_DIR" fetch origin sglang-miles
git -C "$SGLANG_DIR" checkout -q "$SGLANG_PIN"
# Regex-anchored source edits (input_embeds transport/perf, qwen3_5
# ForConditionalGeneration input_embeds bypass). Idempotent; asserts loudly
# if upstream drifted past an anchor.
bash "$NLA_REPO/patches/apply_sglang_patches.sh" "$SGLANG_DIR"

echo "=== [4/8] install sglang (pins torch 2.11 cu130 + transformers 5.8.1) ==="
uv pip install -e "$SGLANG_DIR/python[all]" "${UV_FLAGS[@]}"

echo "=== [5/8] miles checkout @ $MILES_PIN + NLA patches ==="
if [ ! -d "$MILES_DIR/.git" ]; then
    git clone https://github.com/radixark/miles.git "$MILES_DIR"
fi
git -C "$MILES_DIR" fetch origin
# The patch adds --custom-actor-cls-path; its presence marks a patched tree.
if ! grep -q "custom-actor-cls-path" "$MILES_DIR/miles/utils/arguments.py" 2>/dev/null; then
    git -C "$MILES_DIR" checkout -q "$MILES_PIN"
    git -C "$MILES_DIR" apply --whitespace=nowarn "$NLA_REPO"/nla/miles_patches/*.patch
    echo "  applied nla/miles_patches to miles @ $MILES_PIN"
else
    echo "  miles already carries the NLA patches, skipping"
fi
uv pip install -e "$MILES_DIR" "${UV_FLAGS[@]}"

echo "=== [6/8] linear-attention kernels (gated-deltanet layers) ==="
uv pip install flash-linear-attention==0.4.2 "${UV_FLAGS[@]}"
if ! python -c "from causal_conv1d import causal_conv1d_fn" 2>/dev/null; then
    # PyPI wheels target cu12 (libcudart.so.12); build from source against
    # the local CUDA 13 toolkit.
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-32}" \
        uv pip install causal-conv1d --no-binary causal-conv1d --no-build-isolation "${UV_FLAGS[@]}"
fi

echo "=== [7/8] this repo (nla) + HF model ==="
uv pip install -e "$NLA_REPO" "${UV_FLAGS[@]}"
uv pip install -q "huggingface_hub[hf_transfer]"
# Check for actual weights, not just the cache dir — tokenizer-only probes
# create the dir without the safetensors.
if ! ls "$HF_HOME"/hub/models--Qwen--Qwen3.6-27B/snapshots/*/model-00001-of-*.safetensors >/dev/null 2>&1; then
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$BASE_MODEL"
fi

echo "=== [8/8] verify ==="
python - <<'PY'
import torch, transformers, sglang, miles, nla
from transformers import AutoTokenizer
print(f"torch {torch.__version__} (cuda avail: {torch.cuda.is_available()})")
print(f"transformers {transformers.__version__}, sglang {sglang.__version__}")
from causal_conv1d import causal_conv1d_fn  # noqa: F401
from transformers.utils.import_utils import is_flash_linear_attention_available
assert is_flash_linear_attention_available(), "fla missing"
# Tokenizer contract from the dataset sidecar: injection token + neighbors.
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
ids = tok.apply_chat_template(
    [{"role": "user", "content": "<concept>㈜</concept>"}],
    tokenize=True, add_generation_prompt=True, return_dict=False,
)
p = ids.index(158983)
assert (ids[p - 1], ids[p + 1]) == (29, 510), f"neighbor drift: {ids[p-1]}, {ids[p+1]}"
print("tokenizer contract OK (inj 158983, neighbors 29/510)")
PY
for f in "$DATA"/ar_sft_shuf.parquet "$DATA"/av_sft_shuf.parquet "$DATA"/rl_shuf.parquet; do
    [ -f "$f" ] && [ -f "$f.nla_meta.yaml" ] || echo "WARNING: missing $f (+sidecar) — stage the training data before running train_qwen3.6.sh"
done

echo "=== setup complete — next: bash $NLA_REPO/train_qwen3.6.sh ==="
