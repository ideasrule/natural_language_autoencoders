#!/usr/bin/env python
"""Extract last-token residual-stream activations from the base model.

Uses the SAME hook + tokenization as datagen (nla.datagen.extractors.HFExtractor)
so the vectors match what the NLA was trained on: layer_index=K returns the
output of decoder block K (HF hidden_states[K+1]). We keep the activation at the
last real (non-pad) token of each document.

These activations depend only on (base model, layer, input text) — NOT on the
NLA checkpoint — so the output can be reused to evaluate many checkpoints.

Usage:
    python extract_activations.py --base-model Qwen/Qwen2.5-7B-Instruct \
        --layer 20 --input-parquet docs.parquet --text-column content --out acts.parquet
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from nla.datagen.extractors import HFExtractor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--layer", type=int, required=True, help="datagen layer_index K")
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--text-column", default="content")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-rows", type=int, default=None, help="limit rows (debug)")
    args = ap.parse_args()

    tbl = pq.read_table(args.input_parquet)
    texts = tbl.column(args.text_column).to_pylist()
    if args.n_rows:
        texts = texts[: args.n_rows]
    print(f"[extract] {len(texts)} texts; base={args.base_model} layer={args.layer}", flush=True)

    ex = HFExtractor(args.base_model, max_length=args.max_length, batch_size=args.batch_size)
    t0 = time.time()
    results = ex.extract(texts, layer_index=args.layer)
    print(f"[extract] done in {time.time()-t0:.0f}s; {len(results)} results", flush=True)

    # last real token (HFExtractor right-pads/truncates -> [-1] is the last real token)
    vecs = [r.hidden_states[-1].numpy().astype(np.float32) for r in results]
    norms = np.array([float(np.linalg.norm(v)) for v in vecs])
    print(f"[extract] last-token norms: min={norms.min():.1f} max={norms.max():.1f} "
          f"mean={norms.mean():.1f}", flush=True)

    out = pa.table({
        args.text_column: texts,
        "activation_vector": pa.array(vecs, type=pa.list_(pa.float32())),
    })
    pq.write_table(out, args.out)
    print(f"[extract] wrote {args.out} rows={out.num_rows}", flush=True)


if __name__ == "__main__":
    main()
