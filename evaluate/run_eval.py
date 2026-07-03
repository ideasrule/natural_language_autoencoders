#!/usr/bin/env python
"""Generate explanations with the AV and score them with the AR.

For each activation vector:
  AV  (verbalizer): vector -> explanation text   [SGLang input_embeds, NLAClient]
  AR  (reconstructor): explanation -> vector      [NLACritic]
  cosine(AR(explanation), original_vector)        [round-trip fidelity]

Explanation extraction is uniform for every row: pull the <explanation> body
when well-formed; otherwise strip whatever tag scaffolding the AV emitted and
keep the (possibly malformed) body it produced — we never store the raw
tag-wrapped string in the explanation column. The cosine is always computed
from the exact text stored in `explanation`.

Prereq: SGLang already serving the AV HF dir with --disable-radix-cache.

Usage:
    python run_eval.py --av-hf av_hf --ar-hf ar_hf --acts-parquet acts.parquet \
        --out out.parquet [--temperature 0.7] [--text-column content]
"""
import argparse
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from nla_inference import NLAClient, NLACritic  # noqa: E402

_FULL = re.compile(r"<explanation>(.*?)</explanation>", re.S)


def extract_explanation(raw: str) -> str:
    """AV raw output -> explanation body (uniform; tolerant of malformed tags)."""
    raw = raw.strip()
    m = _FULL.search(raw)
    if m:
        return m.group(1).strip()
    # malformed (typo'd opener, or open-without-close): drop one leading <...> tag
    # and one trailing </...> tag, keep the body the AV produced.
    s = re.sub(r"^<[^>]*>\s*", "", raw)
    s = re.sub(r"\s*</[^>]*>\s*$", "", s)
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-hf", required=True, help="AV (actor) HF dir")
    ap.add_argument("--ar-hf", required=True, help="AR (critic) HF dir")
    ap.add_argument("--acts-parquet", required=True, help="parquet with activation_vector column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-column", default="content")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8, help="concurrent AV requests")
    ap.add_argument("--ar-device", default="cuda:1")
    ap.add_argument("--keep-raw", action="store_true",
                    help="also store the AV's raw output in a raw_av_output column")
    args = ap.parse_args()

    acts = pq.read_table(args.acts_parquet)
    vecs = [np.asarray(v, dtype=np.float32) for v in acts.column("activation_vector").to_pylist()]
    texts = (acts.column(args.text_column).to_pylist()
             if args.text_column in acts.schema.names else [None] * len(vecs))
    N = len(vecs)
    print(f"[eval] {N} activations  temp={args.temperature}", flush=True)

    # ---- AV: activation -> raw output (parallel; SGLang batches server-side) ----
    client = NLAClient(args.av_hf, sglang_url=args.sglang_url)

    def gen(i):
        return i, client.generate(vecs[i], extract_explanation=False,
                                  temperature=args.temperature,
                                  max_new_tokens=args.max_new_tokens)
    raw = [None] * N
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, txt in ex.map(gen, range(N)):
            raw[i] = txt
            done += 1
            if done % 100 == 0:
                print(f"[eval] AV {done}/{N} ({time.time()-t0:.0f}s)", flush=True)
    explanations = [extract_explanation(r) for r in raw]
    print(f"[eval] AV done in {time.time()-t0:.0f}s", flush=True)

    # ---- AR: explanation -> reconstructed vector -> cosine vs original ----
    critic = NLACritic(args.ar_hf, device=args.ar_device)
    cos = []
    t0 = time.time()
    for j, (e, v) in enumerate(zip(explanations, vecs)):
        _mse, c = critic.score(e, v)
        cos.append(float(c))
        if (j + 1) % 200 == 0:
            print(f"[eval] AR {j+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[eval] AR done in {time.time()-t0:.0f}s; "
          f"cosine mean={statistics.mean(cos):.4f} "
          f"min={min(cos):.4f} max={max(cos):.4f}", flush=True)

    cols = {args.text_column: texts, "explanation": explanations, "cosine_similarity": cos}
    if args.keep_raw:
        cols["raw_av_output"] = raw
    pq.write_table(pa.table(cols), args.out)
    print(f"[eval] wrote {args.out} rows={N}", flush=True)


if __name__ == "__main__":
    main()
