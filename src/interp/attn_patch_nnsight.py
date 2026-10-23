"""MECH INTERP (causal, sublayer-resolved): patch each layer's ATTENTION output, base <- fine-tuned.

patch_nnsight.py transplants the whole residual stream, so its recovery ramps 0 -> 1 monotonically
BY CONSTRUCTION (a late layer's residual ~ the final residual feeding lm_head). That localizes
poorly. This script instead transplants ONLY the fine-tuned model's self-attention output at a
single layer's last token, leaving the base model's MLPs and residual stream otherwise intact. That
isolates each attention sublayer's MARGINAL causal contribution to the fine-tune's next-token shift
-- and it's the mechanistically right probe, because the LoRA was attention-only (q/k/v/o_proj).
Because only one sublayer's additive contribution is swapped, recovery is NOT monotonic, so a peak
here is a genuine localization (contrast it with the residual-patch ramp).

Metric is the same logit-difference recovery as patch_nnsight.py. Same verified nnsight-0.8
conventions (bare batched layer .output, pre-declared save list, in-place slice writes). Qwen2's
attention module is model.layers[L].self_attn; whether its .output is a bare tensor or an
(attn_out, ...) tuple is DETECTED at runtime (printed as [probe]) and the write branches on it.

Run:  python src/interp/attn_patch_nnsight.py        (GPU node; adapter must already be trained)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_common as C  # noqa: E402

ATTN_IS_TUPLE = None  # set by detect_attn_structure(); a transformers decoder self_attn may return
#                       a bare tensor or a tuple (attn_out, ...), and isinstance() is unreliable on a
#                       live nnsight proxy, so we detect on the MATERIALIZED value and branch on a bool.


def _attn_tensor(layer):
    """The attention-output tensor proxy for a decoder layer (unwrap the tuple if there is one)."""
    out = layer.self_attn.output
    return out[0] if ATTN_IS_TUPLE else out


def detect_attn_structure(lm):
    """One trace: save raw self_attn.output and inspect the real object (proxy isinstance is False)."""
    global ATTN_IS_TUPLE
    with lm.trace(C.PROMPT_TEMPLATE.format(technical=C.EVAL_SAMPLES[0])):
        raw = lm.model.layers[0].self_attn.output.save()
    ATTN_IS_TUPLE = isinstance(raw, (tuple, list))
    h = raw[0] if ATTN_IS_TUPLE else raw
    print(f"[probe] self_attn.output tuple={ATTN_IS_TUPLE} attn.shape={tuple(h.shape)}", flush=True)


def collect_ft_attn(prompts, n_layers):
    """Run FT once; cache per-(prompt,layer) last-position attention output + FT next-token logits."""
    from nnsight import LanguageModel

    ft_hf = C.load_ft(merge=True)
    ft_lm = LanguageModel(ft_hf, tokenizer=C.tokenizer())
    detect_attn_structure(ft_lm)

    per_prompt = []
    for p in prompts:
        attns = []  # declared before the trace so appends survive (see patch_nnsight.py notes)
        with ft_lm.trace(p):
            for L in range(n_layers):
                attns.append(_attn_tensor(ft_lm.model.layers[L])[0, -1, :].save())
            logits = ft_lm.lm_head.output[0, -1, :].save()
        per_prompt.append({
            "attns": [a.detach().float().cpu() for a in attns],
            "logits": logits.detach().float().cpu(),
        })

    C.free(ft_lm, ft_hf)
    return per_prompt


def patch_base(prompts, ft_data, n_layers):
    """Run BASE: baseline logits, then one attention-patched forward per layer. Returns recovery[L]."""
    from nnsight import LanguageModel

    base_hf = C.load_base()
    base_lm = LanguageModel(base_hf, tokenizer=C.tokenizer())

    recov = torch.zeros(n_layers)
    counted = torch.zeros(n_layers)
    top_tokens = []

    tok = C.tokenizer()
    for p, ftd in zip(prompts, ft_data):
        with base_lm.trace(p):
            base_logits = base_lm.lm_head.output[0, -1, :].save()
        base_logits = base_logits.detach().float().cpu()
        ft_logits = ftd["logits"]

        t_star = int(ft_logits.argmax())
        denom = (ft_logits[t_star] - base_logits[t_star]).item()
        top_tokens.append((tok.decode([int(base_logits.argmax())]), tok.decode([t_star])))

        for L in range(n_layers):
            patch = ftd["attns"][L].to(C.DEVICE, dtype=torch.float16)  # [hidden]
            with base_lm.trace(p):
                # in-place slice write of the attention output at the last token; for a tuple output
                # write element [0] in place (reassigning a tuple element raises TypeError in nnsight)
                if ATTN_IS_TUPLE:
                    base_lm.model.layers[L].self_attn.output[0][:, -1, :] = patch
                else:
                    base_lm.model.layers[L].self_attn.output[:, -1, :] = patch
                patched = base_lm.lm_head.output[0, -1, :].save()
            patched = patched.detach().float().cpu()
            if abs(denom) > 1e-4:  # skip prompts where base already agrees with FT (uninformative)
                recov[L] += (patched[t_star] - base_logits[t_star]).item() / denom
                counted[L] += 1

    C.free(base_lm, base_hf)
    counted = counted.clamp(min=1)
    return (recov / counted), top_tokens


def main():
    if C.DEVICE != "cuda":
        sys.exit("Need a GPU node.")
    prompts = [C.PROMPT_TEMPLATE.format(technical=t) for t in C.EVAL_SAMPLES]
    n_layers = C.load_base_config().num_hidden_layers
    print(f"device: {C.DEVICE} | layers: {n_layers} | prompts: {len(prompts)}", flush=True)

    ft_data = collect_ft_attn(prompts, n_layers)
    recovery, top_tokens = patch_base(prompts, ft_data, n_layers)

    lines = ["=" * 72, "ATTENTION-OUTPUT PATCHING  (base <- fine-tuned self_attn output, per layer)", "=" * 72]
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
    lines.append(f"PEAK attention recovery at layer {peak_L} ({recovery[peak_L]:.3f}). Unlike the residual\n"
                 "patch this is NOT monotonic-by-construction, so this peak localizes which attention\n"
                 "sublayer's change most drives the fine-tune's shift (the LoRA was attention-only).")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(os.path.join(C.RESULTS, "attn_patch.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[saved] {os.path.join(C.RESULTS, 'attn_patch.txt')}")


if __name__ == "__main__":
    main()
