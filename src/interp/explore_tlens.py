"""
MECH INTERP (exploratory): can we load Qwen2.5-1.5B into TransformerLens, and what does a
logit lens show?

TransformerLens (`HookedTransformer`) is the most recognizable mech-interp library, but it
supports models via a per-architecture registry — Qwen2.5 may or may not be wired up in the
installed version. This script is deliberately a PROBE: it tries to load the model, and if it
succeeds it runs a classic logit lens (decode each layer's residual stream through the final
norm + unembed, to see which layer the model commits to its top token).

Outcomes we might see, all useful:
  (a) it loads clean         -> we get a logit lens for free, comparable to the nnsight work;
  (b) it loads via a fallback architecture name (Qwen2 config, Qwen2.5 weights);
  (c) it refuses             -> we learn TL can't take Qwen2.5 here and stick with nnsight.
Paste whatever it prints.

Run:  python src/interp/explore_tlens.py            (GPU node)
      python src/interp/explore_tlens.py --ft       (also try the fine-tuned weights)
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_common as C  # noqa: E402

PROBE_PROMPT = C.PROMPT_TEMPLATE.format(technical=C.EVAL_SAMPLES[0])


def try_load(hf_model=None):
    """Attempt HookedTransformer load. Try the real name first, then a Qwen2 fallback name."""
    from transformer_lens import HookedTransformer

    attempts = [C.MODEL, "Qwen/Qwen2-1.5B-Instruct"]  # 2.5 shares Qwen2's architecture
    last_err = None
    for name in attempts:
        try:
            print(f"  trying HookedTransformer.from_pretrained({name!r}, hf_model={'yes' if hf_model else 'no'}) ...", flush=True)
            model = HookedTransformer.from_pretrained(
                name,
                hf_model=hf_model,
                tokenizer=C.tokenizer(),
                device=C.DEVICE,
                dtype=torch.float16,
                fold_ln=False,            # keep it simple/robust for a first probe
                center_writing_weights=False,
                center_unembed=False,
            )
            print(f"  LOADED as {name!r}  (n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model})")
            return model
        except Exception as e:  # noqa: BLE001 - we want to report any failure verbatim
            print(f"  FAILED ({name!r}): {type(e).__name__}: {str(e)[:300]}")
            last_err = e
    raise RuntimeError("TransformerLens could not load Qwen2.5-1.5B in this environment") from last_err


def logit_lens(model, prompt, topk=5):
    """Decode each layer's last-token residual through ln_final + unembed."""
    lines = [f"logit lens — layer-by-layer top-{topk} next-token (prompt ends at 'Plain language:')"]
    _, cache = model.run_with_cache(prompt)
    for L in range(model.cfg.n_layers):
        resid = cache["resid_post", L][:, -1:, :]       # [1,1,d_model]
        logits = model.unembed(model.ln_final(resid))[0, -1]  # [d_vocab]
        top = logits.topk(topk).indices.tolist()
        toks = [repr(model.to_string([t])) for t in top]
        lines.append(f"  L{L:>2}: {', '.join(toks)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ft", action="store_true", help="load fine-tuned (merged) weights instead of base")
    args = ap.parse_args()
    if C.DEVICE != "cuda":
        sys.exit("Need a GPU node.")

    hf_model = None
    if args.ft:
        print("loading fine-tuned (merged) HF weights to hand to TransformerLens ...", flush=True)
        hf_model = C.load_ft(merge=True)

    print(f"=== TransformerLens probe ({'FT' if args.ft else 'BASE'}) ===", flush=True)
    try:
        model = try_load(hf_model=hf_model)
    except RuntimeError as e:
        print("\nRESULT: TransformerLens path unavailable here —", e)
        print("That's a fine outcome: we use nnsight for the mech-interp instead.")
        return

    report = logit_lens(model, PROBE_PROMPT)
    print("\n" + report)
    os.makedirs(C.RESULTS, exist_ok=True)
    tag = "ft" if args.ft else "base"
    out = os.path.join(C.RESULTS, f"tlens_logitlens_{tag}.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
