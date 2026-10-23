"""
MECH INTERP (causal): cross-model activation patching, base <- fine-tuned, with nnsight.

Question: the fine-tune made the model diverge from the base (register switch / hallucination).
WHERE in the network does that divergence become decisive? We test it causally, not by
correlation: take the BASE model, transplant the FINE-TUNED model's residual-stream activation
at one layer (last prompt position), and measure how far the base's next-token prediction moves
toward the fine-tuned model's. The layer where a single-layer transplant "recovers" the FT
decision = where the fine-tune's effect is localized.

Metric (standard activation-patching logit-difference recovery), per layer L:
    t* = argmax of FT's next-token logits (the token FT wants that base doesn't)
    recovery[L] = (logit_patched[t*] - logit_base[t*]) / (logit_ft[t*] - logit_base[t*])
  0.0 = patching layer L changed nothing ; 1.0 = it fully reproduced the FT decision.

We analyze the next-token step right after the 'Plain language:' cue — a single forward pass,
no generation loop. That's deliberately the cleanest patchable probe; it localizes the *onset*
of the divergence, not the whole multi-token hallucination (a later, multi-position extension).

Memory-safe: FT runs first and its activations are cached to CPU, then FT is freed and BASE is
loaded — never both on the GPU at once.

NOTE ON VERSIONS: nnsight's tracing API has changed across releases. This targets nnsight >=0.3
(tested shape of .trace / .save). If the first run throws on .save()/.value or on assigning to a
module output, paste the traceback — the fix is usually a one-line API tweak, not a redesign.

Run:  python src/interp/patch_nnsight.py        (GPU node; adapter must already be trained)
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_common as C  # noqa: E402


def val(proxy):
    """nnsight: a .save()'d proxy exposes its tensor as .value (<=0.3) or acts as it (0.4+)."""
    return proxy.value if hasattr(proxy, "value") else proxy


# Detected once at runtime (see collect_ft_activations) and reused for the base patch, since both
# models share the architecture. Avoids guessing nnsight's output shape: we save the RAW .output
# and inspect the MATERIALIZED tensor, where isinstance/.ndim are reliable (unlike on a proxy).
STRUCT = {"layer_tuple": None, "batched": None}


def slice_last(x):
    """A materialized module output -> its last-token vector [dim].

    Robust to both axes of nnsight/transformers drift: a decoder layer may hand back a tuple
    (hidden, ...) or a bare tensor, and the trace may be batched [b,seq,dim] or unbatched [seq,dim].
    We slice the REAL tensor here (post-trace), never a proxy inside the trace.
    """
    if isinstance(x, (tuple, list)):
        x = x[0]
    return x[-1, :] if x.ndim == 2 else x[0, -1, :]


def collect_ft_activations(prompts, n_layers):
    """Run FT once; cache per-(prompt,layer) last-position residual + FT next-token logits."""
    from nnsight import LanguageModel

    ft_hf = C.load_ft(merge=True)
    ft_lm = LanguageModel(ft_hf, tokenizer=C.tokenizer())

    per_prompt = []
    for p in prompts:
        # Save the WHOLE raw .output (tuple or tensor); slice afterward on the real tensor.
        with ft_lm.trace(p):
            layer_outs = [ft_lm.model.layers[L].output.save() for L in range(n_layers)]
            lm_out = ft_lm.lm_head.output.save()
        layer_vals = [val(o) for o in layer_outs]

        if STRUCT["layer_tuple"] is None:  # detect + announce the output structure exactly once
            lo0 = layer_vals[0]
            STRUCT["layer_tuple"] = isinstance(lo0, (tuple, list))
            h0 = lo0[0] if STRUCT["layer_tuple"] else lo0
            STRUCT["batched"] = h0.ndim == 3
            print(f"[probe] layer_tuple={STRUCT['layer_tuple']} hidden.shape={tuple(h0.shape)} "
                  f"batched={STRUCT['batched']}", flush=True)

        resids_cpu = [slice_last(o).detach().float().cpu() for o in layer_vals]
        logits_cpu = slice_last(val(lm_out)).detach().float().cpu()
        per_prompt.append({"resids": resids_cpu, "logits": logits_cpu})

    C.free(ft_lm, ft_hf)
    return per_prompt


def patch_base(prompts, ft_data, n_layers):
    """Run BASE: baseline logits, then one patched forward per layer. Returns recovery[L] avg."""
    from nnsight import LanguageModel

    base_hf = C.load_base()
    base_lm = LanguageModel(base_hf, tokenizer=C.tokenizer())

    recov = torch.zeros(n_layers)
    counted = torch.zeros(n_layers)
    top_tokens = []  # (base_top_str, ft_top_str) per prompt, to show the divergence

    layer_tuple, batched = STRUCT["layer_tuple"], STRUCT["batched"]  # detected during the FT pass
    tok = C.tokenizer()
    for p, ftd in zip(prompts, ft_data):
        with base_lm.trace(p):
            base_out = base_lm.lm_head.output.save()
        base_logits = slice_last(val(base_out)).detach().float().cpu()
        ft_logits = ftd["logits"]

        t_star = int(ft_logits.argmax())
        denom = (ft_logits[t_star] - base_logits[t_star]).item()
        top_tokens.append((tok.decode([int(base_logits.argmax())]), tok.decode([t_star])))

        for L in range(n_layers):
            patch = ftd["resids"][L].to(C.DEVICE, dtype=torch.float16)
            with base_lm.trace(p):
                out = base_lm.model.layers[L].output       # transplant the FT vector at the last token
                tgt = out[0] if layer_tuple else out        # unwrap proxy tuple via proxy-indexing
                if batched:
                    tgt[0, -1, :] = patch
                else:
                    tgt[-1, :] = patch
                patched_out = base_lm.lm_head.output.save()
            patched = slice_last(val(patched_out)).detach().float().cpu()
            if abs(denom) > 1e-4:  # skip prompts where base already agrees with FT (uninformative)
                recov[L] += (patched[t_star] - base_logits[t_star]).item() / denom
                counted[L] += 1

    C.free(base_lm, base_hf)
    counted = counted.clamp(min=1)
    return (recov / counted), top_tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-samples", type=int, default=len(C.EVAL_SAMPLES))
    args = ap.parse_args()

    if C.DEVICE != "cuda":
        sys.exit("Need a GPU node.")
    prompts = [C.PROMPT_TEMPLATE.format(technical=t) for t in C.EVAL_SAMPLES[: args.max_samples]]
    n_layers = C.load_base_config().num_hidden_layers

    print(f"device: {C.DEVICE} | layers: {n_layers} | prompts: {len(prompts)}", flush=True)
    ft_data = collect_ft_activations(prompts, n_layers)
    recovery, top_tokens = patch_base(prompts, ft_data, n_layers)

    lines = ["=" * 72, "CROSS-MODEL ACTIVATION PATCHING  (base <- fine-tuned residual, per layer)", "=" * 72]
    lines.append("next-token divergence per prompt (base top  ->  FT top):")
    for (bt, ftt), s in zip(top_tokens, C.EVAL_SAMPLES):
        lines.append(f"   {bt!r:>14}  ->  {ftt!r:<14}   | {s[:48]}...")
    lines.append("-" * 72)
    lines.append(f"{'layer':>5} | {'recovery':>9} | bar")
    peak_L = int(recovery.argmax())
    for L, r in enumerate(recovery.tolist()):
        bar = "#" * max(0, int(round(r * 40)))
        star = "  <-- peak" if L == peak_L else ""
        lines.append(f"{L:>5} | {r:>9.3f} | {bar}{star}")
    lines.append("-" * 72)
    lines.append(f"PEAK recovery at layer {peak_L} ({recovery[peak_L]:.3f}) = where the fine-tune's "
                 "next-token decision\nis most causally localized. Compare to the weight-delta "
                 "heatmap (§3) and activation drift (§5).")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(os.path.join(C.RESULTS, "patch_nnsight.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[saved] {os.path.join(C.RESULTS, 'patch_nnsight.txt')}")


if __name__ == "__main__":
    main()
