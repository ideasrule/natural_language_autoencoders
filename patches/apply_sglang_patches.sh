#!/usr/bin/env bash
# Apply NLA patches to an SGLang source tree.
#
# Usage: bash patches/apply_sglang_patches.sh /path/to/sglang
#
# These are regex-anchored edits, not `git apply` patches — SGLang's source
# drifts between releases and exact line numbers don't survive. Each helper
# asserts its anchor matched exactly once; if SGLang has changed enough that
# an anchor misses, the assert tells you which file to patch by hand (the
# sibling *.patch files document the intended diff).
#
# Idempotent: re-running skips already-patched files.

set -euo pipefail

SGLANG_SRC="${1:?usage: $0 /path/to/sglang}"
[ -d "$SGLANG_SRC/python/sglang/srt" ] || {
    echo "error: $SGLANG_SRC does not look like an sglang checkout (no python/sglang/srt)" >&2
    exit 1
}

HTTP="$SGLANG_SRC/python/sglang/srt/entrypoints/http_server.py"
TOK="$SGLANG_SRC/python/sglang/srt/managers/tokenizer_manager.py"
SCHED="$SGLANG_SRC/python/sglang/srt/managers/schedule_batch.py"
GEMMA3="$SGLANG_SRC/python/sglang/srt/models/gemma3_mm.py"
QWEN35="$SGLANG_SRC/python/sglang/srt/models/qwen3_5.py"

# ─── http_server.py: skip FastAPI auto-parse for /generate ─────────────────

_patch_http_server () {
    local f="$1"
    python3 - "$f" <<'PY'
import sys, re
f = sys.argv[1]
s = open(f).read()

if "import dataclasses as _dataclasses" not in s:
    s = re.sub(r"(^import asyncio\n)", r"\1import dataclasses as _dataclasses\n", s, count=1, flags=re.M)

# newer sglang (v0.5.13+) has no plain `import orjson` in http_server.py
if not re.search(r"^import orjson$", s, flags=re.M):
    s = re.sub(r"(^import asyncio\n)", r"\1import orjson\n", s, count=1, flags=re.M)

insert = '''
# === NLA: fields whitelist for manual GenerateReqInput construction ===
# Router injects extra keys (e.g. 'model'); dataclass kwargs are strict.
_GEN_REQ_FIELDS = {f.name for f in _dataclasses.fields(GenerateReqInput)}
'''
if "_GEN_REQ_FIELDS" not in s:
    # decorator is either single-line @app.api_route("/generate", ...) or the
    # v0.5.13+ multi-line form @app.api_route(\n    "/generate",\n    ...)
    s, n_ins = re.subn(
        r'(\n@app\.api_route\((?:\s*\n\s*)?"/generate")', insert + r"\1", s, count=1
    )
    assert n_ins == 1, "could not find /generate route decorator to anchor _GEN_REQ_FIELDS"

handler_pat = re.compile(
    r'(@app\.api_route\(\s*\n?\s*"/generate",\s*\n?\s*methods=\["POST", "PUT"\],?\s*\n?'
    r'(?:\s*response_class=\w+,?\s*\n?)?'
    r'\s*\)\s*\n'
    r'(?:@\w.*\n)*'
    r')async def generate_request\(obj: GenerateReqInput, request: Request\):\s*\n'
    r'(\s*)"""Handle a generate request\."""\s*\n'
)
handler_sub = r'''\1async def generate_request(request: Request):
\2"""Handle a generate request."""
\2# === NLA: skip FastAPI auto-parse (155ms/req for 448K floats) ===
\2data = orjson.loads(await request.body())
\2obj = GenerateReqInput(**{k: v for k, v in data.items() if k in _GEN_REQ_FIELDS})
'''
s, n = handler_pat.subn(handler_sub, s)
assert n == 1, f"generate handler pattern matched {n} times (expected 1) — sglang source changed, patch manually"

open(f, "w").write(s)
print(f"  patched {f}")
PY
}

# ─── http_server.py: bf16-base64 input_embeds transport ────────────────────

_patch_http_server_b64 () {
    local f="$1"
    python3 - "$f" <<'PY'
import sys, re
f = sys.argv[1]
s = open(f).read()

# Anchor on the line _patch_http_server just inserted.
pat = re.compile(
    r'([ \t]+)data = orjson\.loads\(await request\.body\(\)\)\n'
)
sub = r'''\1data = orjson.loads(await request.body())
\1# === NLA: bf16-base64 input_embeds (12MB->2.8MB on the wire) ===
\1# Client sends bf16 bytes (viewed as int16; numpy has no bf16) b64-encoded
\1# + shape; reinterpret here. Downstream io_struct.py isinstance(..., float)
\1# needs Python floats so .tolist(). schedule_batch casts to bf16 anyway,
\1# so bf16 transport is bit-exact end-to-end.
\1if "input_embeds_b64_bf16" in data:
\1    import base64, numpy as np, torch
\1    raw = base64.b64decode(data.pop("input_embeds_b64_bf16"))
\1    shape = data.pop("input_embeds_shape")
\1    i16 = np.frombuffer(raw, dtype=np.int16).reshape(shape).copy()
\1    data["input_embeds"] = (
\1        torch.from_numpy(i16).view(torch.bfloat16).float().tolist()
\1    )
'''
s, n = pat.subn(sub, s, count=1)
assert n == 1, f"b64 anchor matched {n} times — run _patch_http_server first, or patch manually"
open(f, "w").write(s)
print(f"  patched {f} (b64)")
PY
}

# ─── tokenizer_manager.py: nested-list → ndarray before pickle ─────────────

_patch_tokenizer () {
    local f="$1"
    python3 - "$f" <<'PY'
import sys, re
f = sys.argv[1]
s = open(f).read()

# `[ \t]+` not `\s+` — the latter eats the preceding \n and every `\1` in
# the sub re-emits it, producing spurious blank lines.
pat = re.compile(r'([ \t]+)input_embeds = obj\.input_embeds(\s*\n)')
sub = r'''\1# === NLA: numpy conversion before pickle ===
\1# Nested list[list[float]] (448K PyFloat) -> ndarray. pickle 9->2.6ms,
\1# unpickle 17->1ms, torch.tensor 83->0.1ms (from_numpy zero-copy).
\1import numpy as np
\1input_embeds = np.asarray(obj.input_embeds, dtype=np.float32)\2'''
s, n = pat.subn(sub, s, count=1)
assert n == 1, f"tokenizer input_embeds pattern matched {n} times — check manually"
open(f, "w").write(s)
print(f"  patched {f}")
PY
}

# ─── schedule_batch.py: append+concat ndarrays; clear embeds on decode ─────

_patch_schedule () {
    local f="$1"
    python3 - "$f" <<'PY'
import sys, re
f = sys.argv[1]
s = open(f).read()

# 1. .extend → .append (req.input_embeds is now an ndarray, keep it whole).
# Upstream (v0.5.13+) already slices to the chunked-prefill window
# (req.input_embeds[pre_len : pre_len + req.extend_input_len]) — keep the
# slice, just swap extend→append so the ndarray stays 2-D for the concat.
pat1 = re.compile(
    r'([ \t]+)input_embeds\.extend\(\n'
    r'[ \t]+(req\.input_embeds\[pre_len : pre_len \+ req\.extend_input_len\])\n'
    r'[ \t]+\)'
)
sub1 = r'''\1# === NLA: append ndarray (from tokenizer_manager), concat later ===
\1input_embeds.append(\2)'''
s, n1 = pat1.subn(sub1, s, count=1)
if n1 != 1:
    # pre-v0.5.13 layout: unsliced extend on one line
    pat1b = re.compile(
        r'(\s+if req\.input_embeds is not None:\s*\n)'
        r'(\s+#[^\n]*\n)?'
        r'(\s+)input_embeds\.extend\(req\.input_embeds\)[^\n]*'
    )
    sub1b = r'''\1\3# === NLA: append ndarray (from tokenizer_manager), concat later ===
\3input_embeds.append(req.input_embeds)'''
    s, n1b = pat1b.subn(sub1b, s, count=1)
    assert n1b == 1, "extend->append pattern matched neither new nor old layout — patch manually"

# 2. torch.tensor(nested_list) → from_numpy(concatenate), cast bf16
pat2 = re.compile(
    r'self\.input_embeds = \(\s*\n'
    r'\s+torch\.tensor\(input_embeds(?:, pin_memory=\w+)?\)\.to\(\s*\n?'
    r'\s*self\.device, non_blocking=True\s*\n?'
    r'\s*\)\s*\n'
    r'\s+if input_embeds\s*\n'
    r'\s+else None\s*\n'
    r'\s+\)'
)
sub2 = '''import numpy as np
        self.input_embeds = (
            torch.from_numpy(
                np.concatenate([np.asarray(e, dtype=np.float32) for e in input_embeds])
            ).to(self.device, dtype=torch.bfloat16, non_blocking=True)
            if input_embeds
            else None
        )'''
s, n2 = pat2.subn(sub2, s, count=1)
assert n2 == 1, f"torch.tensor->from_numpy pattern matched {n2} times — sglang source changed, patch manually"

# 3. prepare_for_decode: clear stale prompt embeds (decode embeds last_output_id)
pat3 = re.compile(
    r'([ \t]+)def prepare_for_decode\(self\):\n'
    r'([ \t]+)self\.forward_mode = ForwardMode\.DECODE\n'
)
sub3 = r'''\1def prepare_for_decode(self):
\2self.forward_mode = ForwardMode.DECODE
\2# === NLA: clear stale embeds before decode ===
\2self.input_embeds = None
'''
# Skip ONLY if upstream already clears input_embeds UNCONDITIONALLY: the
# assignment must sit at the same indent as the method body's first statement
# (a deeper indent means it's inside an if/for — conditional, not equivalent),
# within the first few statements, with no intervening dedent out of the method.
_upstream_clear = re.compile(
    r"def prepare_for_decode\(self\):\n"
    r"([ \t]+)\S[^\n]*\n"                      # first body line captures body indent
    r"(?:\1[^\n]*\n|[ \t]*#[^\n]*\n|\n){0,8}?"  # same-indent lines / comments / blanks only
    r"\1self\.input_embeds = None\n"
)
if _upstream_clear.search(s):
    print("  prepare_for_decode already clears input_embeds unconditionally (upstream), skipping hunk 3")
    n3 = 1
else:
    s, n3 = pat3.subn(sub3, s, count=1)
assert n3 == 1, f"prepare_for_decode pattern matched {n3} times — sglang source changed, patch manually"

open(f, "w").write(s)
print(f"  patched {f}")
PY
}

# ─── schedule_batch.py: reset_for_retract must drop output_ids/logprobs ────

_patch_retract () {
    local f="$1"
    python3 - "$f" <<'PY'
import sys, re
f = sys.argv[1]
s = open(f).read()

# Anchor on the last two lines of reset_for_retract before the next method.
pat = re.compile(
    r'([ \t]+)self\.kv_committed_freed = False\n'
    r'[ \t]+self\.kv_overallocated_freed = False\n'
    r'\n'
    r'([ \t]+)def offload_kv_cache\('
)
sub = r'''\1self.kv_committed_freed = False
\1self.kv_overallocated_freed = False

\1# Upstream PR #14110: input_embeds reqs can't mix original embeddings
\1# with output token IDs on re-prefill. Keeping output_ids -> scheduler
\1# allocates len(embeds)+len(output_ids) KV slots but model only fills
\1# len(embeds) -> shape mismatch. Discard progress, restart from zero.
\1if self.input_embeds is not None:
\1    self.output_ids = []
\1    if self.return_logprob:
\1        self.output_token_logprobs_val = []
\1        self.output_token_logprobs_idx = []
\1        self.output_top_logprobs_val = []
\1        self.output_top_logprobs_idx = []
\1        self.output_token_ids_logprobs_val = []
\1        self.output_token_ids_logprobs_idx = []
\1    self.hidden_states = []

\2def offload_kv_cache('''
s, n = pat.subn(sub, s, count=1)
assert n == 1, f"retract fix anchor matched {n} times — sglang source changed, patch manually"
open(f, "w").write(s)
print(f"  patched {f} (retract fix)")
PY
}

# ─── schedule_batch.py: slice input_embeds to chunked-prefill window ───────

_patch_chunked_prefill () {
    local f="$1"
    python3 - "$f" <<'PY'
import sys, re
f = sys.argv[1]
s = open(f).read()

# Anchors on the line _patch_schedule inserted.
pat = re.compile(
    r'([ \t]+)if req\.input_embeds is not None:\n'
    r'[ \t]+# === NLA: append ndarray \(from tokenizer_manager\), concat later ===\n'
    r'[ \t]+input_embeds\.append\(req\.input_embeds\)\n'
)
sub = (
    r'\1if req.input_embeds is not None:\n'
    r'\1    # === NLA: append ndarray (from tokenizer_manager), concat later ===\n'
    r'\1    # Slice to match chunked-prefill truncation of fill_ids.\n'
    r'\1    # pre_len = len(prefix_indices) = rows consumed in prior chunks.\n'
    r'\1    # req.extend_input_len = this chunk budget (adder-truncated).\n'
    r'\1    # Non-chunked: pre_len=0, extend_input_len=len(embeds) -> full slice.\n'
    r'\1    input_embeds.append(\n'
    r'\1        req.input_embeds[pre_len : pre_len + req.extend_input_len]\n'
    r'\1    )\n'
)
s, n = pat.subn(sub, s, count=1)
assert n == 1, f"chunked-prefill fix anchor matched {n} times — run _patch_schedule first, or patch manually"
open(f, "w").write(s)
print(f"  patched {f} (chunked-prefill fix)")
PY
}

# ─── gemma3_mm.py: route input_embeds straight to language_model ───────────

_patch_gemma3_mm () {
    local f="$1"
    python3 - "$f" <<'PY'
import sys, re
f = sys.argv[1]
s = open(f).read()

pat = re.compile(
    r'([ \t]+# Important: position_ids in Gemma3 are 1-indexed\n'
    r'[ \t]+# This really does cost me sometime\n'
    r'([ \t]+)positions \+= 1\n)'
)
sub = r'''\1\2# === NLA: input_embeds bypass — text-only injection path ===
\2# general_mm_embed_routine only reads input_ids, so injected embeds
\2# would be silently dropped. When input_embeds is provided, call the
\2# language_model directly (Gemma3ForCausalLM handles input_embeds).
\2if input_embeds is not None:
\2    return self.language_model(input_ids, positions, forward_batch, input_embeds, **kwargs)
'''
s, n = pat.subn(sub, s, count=1)
assert n == 1, f"gemma3_mm anchor matched {n} times — sglang source changed, patch manually"
open(f, "w").write(s)
print(f"  patched {f}")
PY
}


# ─── qwen3_5.py: route input_embeds straight to the language model ─────────

_patch_qwen3_5_mm () {
    local f="$1"
    python3 - "$f" <<'PY'
import sys, re
f = sys.argv[1]
s = open(f).read()

# Insert a forward override into Qwen3_5ForConditionalGeneration (dense).
# The inherited Qwen3VLForConditionalGeneration.forward has no input_embeds
# kwarg -> TypeError on the NLA injection path. Qwen3_5ForCausalLM.forward
# (self.model) accepts input_embeds and returns hidden states; replicate the
# wrapper's logits step on top.
pat = re.compile(
    r'([ \t]+)self\.deepstack_visual_indexes = self\.visual\.deepstack_visual_indexes\n'
    r'\n'
    r'([ \t]+)def get_hidden_dim\(self, module_name: str, layer_idx: int\):\n'
    r'([ \t]+)return self\.model\.get_hidden_dim\(module_name, layer_idx\)\n'
)
sub = r"""\1self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes

\2# === NLA: input_embeds bypass — text-only injection path ===
\2# general_mm_embed_routine only reads input_ids, so injected embeds
\2# would be silently dropped (and the inherited forward rejects the
\2# kwarg). When input_embeds is provided, call the language model
\2# directly (Qwen3_5ForCausalLM.forward handles input_embeds and
\2# returns hidden states), then apply the logits step.
\2@torch.no_grad()
\2def forward(
\2    self,
\2    input_ids,
\2    positions,
\2    forward_batch,
\2    input_embeds=None,
\2    get_embedding: bool = False,
\2    pp_proxy_tensors=None,
\2):
\2    if input_embeds is None:
\2        return super().forward(
\2            input_ids, positions, forward_batch,
\2            get_embedding=get_embedding, pp_proxy_tensors=pp_proxy_tensors,
\2        )
\2    assert not get_embedding
\2    if self.is_mrope_enabled:
\2        positions = forward_batch.mrope_positions
\2    hidden_states = self.model(
\2        input_ids, positions, forward_batch, input_embeds,
\2        pp_proxy_tensors=pp_proxy_tensors,
\2    )
\2    aux_hidden_states = None
\2    if self.capture_aux_hidden_states:
\2        hidden_states, aux_hidden_states = hidden_states
\2    if self.pp_group.is_last_rank:
\2        return self.logits_processor(
\2            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
\2        )
\2    return hidden_states
\2# === end NLA patch ===

\2def get_hidden_dim(self, module_name: str, layer_idx: int):
\3return self.model.get_hidden_dim(module_name, layer_idx)
"""
s, n = pat.subn(sub, s, count=1)
assert n == 1, f"qwen3_5 CondGen anchor matched {n} times — sglang source changed, patch manually"
open(f, "w").write(s)
print(f"  patched {f} (input_embeds bypass)")
PY
}

# ─── orchestrate ───────────────────────────────────────────────────────────

echo "=== applying NLA SGLang patches to $SGLANG_SRC ==="

if grep -q "_GEN_REQ_FIELDS" "$HTTP"; then
    echo "  http_server.py already patched, skipping"
else
    _patch_http_server "$HTTP"
fi

if grep -q "input_embeds_b64_bf16" "$HTTP"; then
    echo "  http_server.py b64 already patched, skipping"
else
    _patch_http_server_b64 "$HTTP"
fi

if grep -q "numpy conversion before pickle" "$TOK"; then
    echo "  tokenizer_manager.py already patched, skipping"
else
    _patch_tokenizer "$TOK"
fi

if grep -q "np\.concatenate.*for e in input_embeds" "$SCHED"; then
    echo "  schedule_batch.py already patched, skipping"
else
    _patch_schedule "$SCHED"
fi

# Retract fix (PR #14110) was merged upstream in v0.5.13+: reset_for_retract
# discards output_ids for input_embeds reqs ("we discard the generated
# output_ids and restart prefill"). Only patch older checkouts.
if grep -q "Upstream PR #14110" "$SCHED" || grep -q "discard the generated output_ids and restart prefill" "$SCHED"; then
    echo "  schedule_batch.py retract fix present (patched or upstream), skipping"
else
    _patch_retract "$SCHED"
fi

# Chunked-prefill slice also merged upstream in v0.5.13+ (PrefillAdder comment
# + [pre_len : pre_len + req.extend_input_len] slice). Only patch older checkouts.
if grep -q "Slice to match chunked-prefill" "$SCHED" || grep -q "pre_len : pre_len + req.extend_input_len" "$SCHED"; then
    echo "  schedule_batch.py chunked-prefill fix present (patched or upstream), skipping"
else
    _patch_chunked_prefill "$SCHED"
fi

if [ -f "$GEMMA3" ]; then
    if grep -q "NLA: input_embeds bypass" "$GEMMA3"; then
        echo "  gemma3_mm.py already patched, skipping"
    else
        _patch_gemma3_mm "$GEMMA3"
    fi
else
    echo "  gemma3_mm.py not present in this sglang version, skipping"
fi

if [ -f "$QWEN35" ]; then
    if grep -q "NLA: input_embeds bypass" "$QWEN35"; then
        echo "  qwen3_5.py already patched, skipping"
    else
        _patch_qwen3_5_mm "$QWEN35"
    fi
else
    echo "  qwen3_5.py not present in this sglang version, skipping"
fi

echo "=== verifying ==="
grep -q "_GEN_REQ_FIELDS" "$HTTP"                          && echo "  ok http_server.py"
grep -q "input_embeds_b64_bf16" "$HTTP"                    && echo "  ok http_server.py (b64)"
grep -q "numpy conversion before pickle" "$TOK"            && echo "  ok tokenizer_manager.py"
grep -q "np\.concatenate.*for e in input_embeds" "$SCHED"  && echo "  ok schedule_batch.py (perf)"
grep -q "Upstream PR #14110" "$SCHED"                      && echo "  ok schedule_batch.py (retract)"
grep -q "Slice to match chunked-prefill" "$SCHED"          && echo "  ok schedule_batch.py (chunked-prefill)"
[ ! -f "$GEMMA3" ] || grep -q "NLA: input_embeds bypass" "$GEMMA3" && echo "  ok gemma3_mm.py"
[ ! -f "$QWEN35" ] || grep -q "NLA: input_embeds bypass" "$QWEN35" && echo "  ok qwen3_5.py"
echo "=== done ==="
