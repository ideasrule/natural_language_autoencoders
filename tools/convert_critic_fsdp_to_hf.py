#!/usr/bin/env python
"""Convert an RL/SFT critic (AR) FSDP-DCP checkpoint -> HF format.

The critic is an NLACriticModel = truncated backbone + Linear value_head. The
generic tools/convert_fsdp_to_hf.py can't handle it: it builds a plain
CausalLM skeleton, so the `value_head` and the `backbone.` key prefix don't
map. Instead we build an NLACriticModel skeleton and load the DCP state into
it, then NLACriticModel.save_pretrained writes the backbone HF shards AND
value_head.safetensors.

Everything is derived from the checkpoint — base model from the sidecar, layer
count K from the DCP's own layer keys — so this works for any model family /
extraction layer, not just Qwen-L20.

Usage:
    python convert_critic_to_hf.py --critic-dcp .../critic/iter_0000400 \
        --out .../ar_hf [--base-model <hf id/dir>]
"""
import argparse
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from nla.models import NLACriticModel  # noqa: E402


def _load_convert_helpers():
    """Borrow the no-dist DCP state-dict loader from tools/convert_fsdp_to_hf.py."""
    spec = importlib.util.spec_from_file_location(
        "_cvt", str(REPO / "tools" / "convert_fsdp_to_hf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _derive_base_model(critic_dcp: Path, override: str | None) -> str:
    if override:
        return override
    meta = critic_dcp / "nla_meta.yaml"
    assert meta.exists(), (
        f"no nla_meta.yaml in {critic_dcp} — pass --base-model explicitly")
    m = yaml.safe_load(meta.read_text())
    base = m.get("base_checkpoint") or (m.get("parent_checkpoints") or [None])[0]
    assert base, f"could not find base model in {meta}; pass --base-model"
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic-dcp", required=True, help="critic FSDP-DCP iter dir (has model/)")
    ap.add_argument("--out", required=True, help="output HF dir")
    ap.add_argument("--base-model", default=None,
                    help="base HF model for the skeleton (default: read from sidecar)")
    args = ap.parse_args()

    critic_dcp = Path(args.critic_dcp)
    out = Path(args.out)
    base_model = _derive_base_model(critic_dcp, args.base_model)
    cvt = _load_convert_helpers()

    print(f"[critic->hf] loading DCP {critic_dcp} (no_dist)", flush=True)
    raw = cvt._load_fsdp_state_dict(cvt._detect_model_dir(str(critic_dcp)))

    # K (= datagen extraction layer): blocks 0..K kept -> max layer index in the DCP.
    layer_ids = [int(m.group(1)) for k in raw
                 for m in [re.search(r"\.layers\.(\d+)\.", k)] if m]
    assert layer_ids, "no decoder-layer keys found in critic DCP"
    K = max(layer_ids)
    print(f"[critic->hf] base_model={base_model}  K={K} (num_hidden_layers={K+1})", flush=True)

    print("[critic->hf] building NLACriticModel skeleton", flush=True)
    model = NLACriticModel.from_pretrained(
        base_model, nla_num_layers=K, torch_dtype=torch.float32, attn_implementation="sdpa")
    tgt = set(model.state_dict().keys())

    # pick the DCP key prefix (FSDP ModelState wrapping) that best matches the model
    def strip(d, p):
        return {k[len(p):] if k.startswith(p) else k: v for k, v in d.items()}
    best_p, best_n = "", -1
    for p in ("model_state.model.", "model_state.", "module.", "model.", ""):
        n = len(set(strip(raw, p)) & tgt)
        if n > best_n:
            best_p, best_n = p, n
    print(f"[critic->hf] prefix={best_p!r} matched {best_n}/{len(tgt)}", flush=True)
    sd = strip(raw, best_p)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"incomplete load, missing: {list(missing)[:8]}"
    vh = model.value_head.weight
    eye_dist = torch.norm(vh.float() - torch.eye(vh.shape[0])).item()
    print(f"[critic->hf] loaded; value_head ||W-I||={eye_dist:.3f} "
          f"(0 == identity/untrained)", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out)

    # tokenizer (from base model) + sidecar (mse_scale + critic prompt template)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(out)
    shutil.copy(critic_dcp / "nla_meta.yaml", out / "nla_meta.yaml")
    print(f"[critic->hf] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
