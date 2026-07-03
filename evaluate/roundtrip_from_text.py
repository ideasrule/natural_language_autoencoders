#!/usr/bin/env python
"""End-to-end NLA round-trip eval starting from RAW TEXT snippets.

Unlike run_eval.py (which starts from a parquet of precomputed activation
vectors and needs a running SGLang server), this script does the whole loop
in-process with plain HF forward passes — no server, no precomputed
activations:

  1. base model forward on `content` → layer-K activation at the LAST token
     (datagen convention: layer_index=K is the output of block K, i.e. HF
     hidden_states[K+1]; tokenized with add_special_tokens=True to match the
     stage-0 extractor)
  2. the AV generates an explanation for that vector — sidecar prompt
     template, sidecar injection_scale, thinking disabled, greedy decoding
  3. the AR maps the explanation back to a vector (suffix-anchored last token)
  4. cosine_similarity(reconstruction, original)

Output parquet columns: content, explanation, cosine_similarity.

Batching note: every AV prompt is the SAME token sequence (only the injected
vector differs), so generation is batched with NO padding. This is
load-bearing for hybrid linear-attention models (Qwen3.5): the gated-deltanet
recurrent scan does not consult attention_mask, so left-padding would pollute
the state. Extraction batches are right-padded, which is causal-safe.

Usage (defaults reproduce the Qwen3.5-9B eval):
    python evaluate/roundtrip_from_text.py \
        --test-parquet /workspace/data/test_data_ultrafineweb.parquet \
        --out /workspace/data/test_results.parquet \
        --base-model Qwen/Qwen3.5-9B \
        --av-hf /workspace/runs/rl_actor_hf \
        --av-sidecar /workspace/runs/rl/actor/iter_0000780 \
        --ar-hf /workspace/runs/rl/critic/iter_0000780/hf

The AV checkpoint must be an HF export (actor RL saves are DCP — convert with
tools/convert_fsdp_to_hf.py first). --av-sidecar points at the dir whose
nla_meta.yaml carries the injection_scale the AV was trained with.
"""
import argparse
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.injection import inject_at_marked_positions
from nla.models import NLACriticModel
from nla.schema import extract_explanation, normalize_activation


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-parquet", required=True, help="parquet with a text column to explain")
    p.add_argument("--out", required=True, help="output parquet (content/explanation/cosine_similarity)")
    p.add_argument("--base-model", required=True, help="HF checkpoint activations are extracted from")
    p.add_argument("--av-hf", required=True, help="AV (actor) HF checkpoint dir")
    p.add_argument("--av-sidecar", required=True,
                   help="dir whose nla_meta.yaml has the AV's injection_scale (e.g. the actor iter dir)")
    p.add_argument("--ar-hf", required=True, help="AR (critic) HF checkpoint dir")
    p.add_argument("--text-column", default="content")
    p.add_argument("--layer", type=int, default=21,
                   help="extraction layer_index K (output of block K = hidden_states[K+1])")
    p.add_argument("--max-content-tokens", type=int, default=1024,
                   help="clip snippets; the 'last token' is the last token of the clipped snippet")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--batch-extract", type=int, default=16)
    p.add_argument("--batch-av", type=int, default=16)
    p.add_argument("--batch-ar", type=int, default=32)
    p.add_argument("--extract-device", default="cuda:0")
    p.add_argument("--av-device", default="cuda:1")
    p.add_argument("--ar-device", default="cuda:2")
    return p.parse_args()


def extract_activations(args, tok, rows):
    """Base-model layer-K activation at each snippet's last real token. [N, d] raw."""
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16).to(args.extract_device).eval()
    acts = []
    t0 = time.time()
    for i in range(0, len(rows), args.batch_extract):
        chunk = rows[i : i + args.batch_extract]
        enc = tok(chunk, add_special_tokens=True, truncation=True,
                  max_length=args.max_content_tokens, padding=True,
                  return_tensors="pt").to(args.extract_device)
        with torch.no_grad():
            hs = base(**enc, output_hidden_states=True, use_cache=False
                      ).hidden_states[args.layer + 1]
        last = enc["attention_mask"].cumsum(1).argmax(1)
        acts.append(hs[torch.arange(hs.shape[0], device=hs.device), last].float().cpu())
        if i % (10 * args.batch_extract) == 0:
            print(f"  extract {i}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    del base
    torch.cuda.empty_cache()
    acts = torch.cat(acts)
    print(f"extraction done: {tuple(acts.shape)}, mean raw norm {acts.norm(dim=1).mean():.1f}", flush=True)
    return acts


def generate_explanations(args, tok, cfg, acts):
    """AV explanations, greedy, thinking disabled. Falls back to the raw
    generation (tags stripped by extract_explanation upstream) if the
    <explanation> block is malformed."""
    av = AutoModelForCausalLM.from_pretrained(
        args.av_hf, dtype=torch.bfloat16).to(args.av_device).eval()
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
        enable_thinking=False,
    )
    prompt = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(args.av_device)
    print(f"AV prompt: {len(prompt_ids)} tokens", flush=True)

    holder = {}

    def hook(_m, inputs, output):
        v = holder.get("v")
        # Decode steps carry a single token — no marker to inject into.
        if v is None or inputs[0].shape[-1] == 1:
            return output
        return inject_at_marked_positions(
            inputs[0], output, v,
            cfg.injection_token_id, cfg.injection_left_neighbor_id,
            cfg.injection_right_neighbor_id,
        )

    handle = av.get_input_embeddings().register_forward_hook(hook)
    explanations = []
    t0 = time.time()
    for i in range(0, acts.shape[0], args.batch_av):
        n = min(args.batch_av, acts.shape[0] - i)
        holder["v"] = normalize_activation(acts[i : i + n], cfg.injection_scale).to(args.av_device)
        with torch.no_grad():
            out = av.generate(
                prompt.expand(n, -1), max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        for j in range(n):
            text = tok.decode(out[j, prompt.shape[1]:], skip_special_tokens=True)
            expl = extract_explanation(text)
            explanations.append(expl if expl is not None else text.strip())
        print(f"  AV {i+n}/{acts.shape[0]} ({time.time()-t0:.0f}s)", flush=True)
    handle.remove()
    del av
    torch.cuda.empty_cache()
    return explanations


def reconstruct(args, cfg, explanations):
    """AR forward on the critic template; prediction at the suffix-anchored last token."""
    ar_tok = AutoTokenizer.from_pretrained(args.ar_hf)
    ar_tok.padding_side = "right"
    ar = NLACriticModel.from_pretrained(
        args.ar_hf, torch_dtype=torch.bfloat16).to(args.ar_device).eval()
    preds = []
    for i in range(0, len(explanations), args.batch_ar):
        prompts = [cfg.critic_prompt_template.format(explanation=e)
                   for e in explanations[i : i + args.batch_ar]]
        enc = ar_tok(prompts, add_special_tokens=True, padding=True, truncation=True,
                     max_length=384, return_tensors="pt").to(args.ar_device)
        last = enc["attention_mask"].cumsum(1).argmax(1)
        with torch.no_grad():
            v = ar(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                   use_cache=False).values
        preds.append(v[torch.arange(v.shape[0], device=v.device), last].float().cpu())
    del ar
    torch.cuda.empty_cache()
    return torch.cat(preds)


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    cfg = load_nla_config(args.av_sidecar, tok)
    assert cfg.injection_scale is not None, (
        f"{args.av_sidecar} sidecar carries no injection_scale — point --av-sidecar "
        f"at the trained AV's iter dir (model sidecars bake the trained scale)."
    )
    assert cfg.critic_prompt_template is not None
    print(f"sidecar: injection_scale={cfg.injection_scale}, inj_id={cfg.injection_token_id}", flush=True)

    rows = pq.read_table(args.test_parquet, columns=[args.text_column]
                         ).column(args.text_column).to_pylist()
    print(f"{len(rows)} test snippets", flush=True)

    acts = extract_activations(args, tok, rows)
    explanations = generate_explanations(args, tok, cfg, acts)
    preds = reconstruct(args, cfg, explanations)

    cos = torch.nn.functional.cosine_similarity(preds, acts, dim=1).numpy()
    print(f"cosine: mean={cos.mean():.4f} median={np.median(cos):.4f} "
          f"p10={np.percentile(cos,10):.4f} p90={np.percentile(cos,90):.4f} "
          f"min={cos.min():.4f}", flush=True)
    print(f"(unit-vector mse equivalent 2(1-cos): {2*(1-cos.mean()):.4f})", flush=True)

    pq.write_table(
        pa.table({
            "content": pa.array(rows, type=pa.string()),
            "explanation": pa.array(explanations, type=pa.string()),
            "cosine_similarity": pa.array(cos.astype(np.float64)),
        }),
        args.out,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
