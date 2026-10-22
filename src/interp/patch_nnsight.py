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


def collect_ft_activations(prompts, n_layers):
    """Run FT once; cache per-(prompt,layer) last-position residual + FT next-token logits."""
    from nnsight import LanguageModel

    ft_hf = C.load_ft(merge=True)
    ft_lm = LanguageModel(ft_hf, tokenizer=C.tokenizer())

    per_prompt = []
    for p in prompts:
        with ft_lm.trace(p):
            resids = [ft_lm.model.layers[L].output[0][0, -1, :].save() for L in range(n_layers)]
            logits = ft_lm.lm_head.output[0, -1, :].save()
        resids_cpu = [val(r).detach().float().cpu() for r in resids]
        per_prompt.append({"resids": resids_cpu, "logits": val(logits).detach().float().cpu()})

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

    tok = C.tokenizer()
    for p, ftd in zip(prompts, ft_data):
        with base_lm.trace(p):
            base_logits = base_lm.lm_head.output[0, -1, :].save()
        base_logits = val(base_logits).detach().float().cpu()
        ft_logits = ftd["logits"]

        t_star = int(ft_logits.argmax())
        denom = (ft_logits[t_star] - base_logits[t_star]).item()
        top_tokens.append((tok.decode([int(base_logits.argmax())]), tok.decode([t_star])))

        for L in range(n_layers):
            patch = ftd["resids"][L].to(C.DEVICE, dtype=torch.float16)
            with base_lm.trace(p):
                base_lm.model.layers[L].output[0][0, -1, :] = patch
                patched = base_lm.lm_head.output[0, -1, :].save()
            patched = val(patched).detach().float().cpu()
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
