"""LOGIT LENS (base vs fine-tuned) via nnsight — the fast replacement for the TransformerLens probe.

A logit lens is just "decode the residual stream early". For each layer L we take the last-token residual, push it through the model's OWN
final norm + lm_head, and read the top token — i.e. what the model would predict if it stopped at L.

Comparing the base vs the fine-tuned trajectory shows WHERE each model's next-token decision
crystallizes, and at which depth the fine-tune locks onto its review-summary register ('We' ...)
while the base stays input-faithful. Triangulates with the activation-patching late-layer ramp.

Uses the verified nnsight-0.8 pattern from patch_nnsight.py (bare batched .output, pre-declared save
list). Memory-safe: base and FT are loaded one at a time. Run:
    python src/interp/logit_lens_nnsight.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_common as C  # noqa: E402


def lens_trajectory(load_model, prompts, n_layers):
    """Per prompt, the logit-lens top token id at each layer (last prompt position)."""
    from nnsight import LanguageModel

    hf = load_model()
    lm = LanguageModel(hf, tokenizer=C.tokenizer())
    dev = next(hf.parameters()).device
    dt = next(hf.parameters()).dtype
    norm, head = hf.model.norm, hf.lm_head  # Qwen2: final RMSNorm then the LM head

    per_prompt = []
    for p in prompts:
        resids = []  # declare BEFORE the trace (nnsight exec's the body in its own locals)
        with lm.trace(p):
            for L in range(n_layers):
                resids.append(lm.model.layers[L].output[0, -1, :].save())
        tops = []
        with torch.no_grad():
            for r in resids:
                logits_L = head(norm(r.detach().to(device=dev, dtype=dt)))
                tops.append(int(logits_L.argmax()))
        per_prompt.append(tops)

    C.free(lm, hf)
    return per_prompt


def main():
    if C.DEVICE != "cuda":
        sys.exit("Need a GPU node.")
    tok = C.tokenizer()
    prompts = [C.PROMPT_TEMPLATE.format(technical=t) for t in C.EVAL_SAMPLES]
    n_layers = C.load_base_config().num_hidden_layers
    print(f"device: {C.DEVICE} | layers: {n_layers} | prompts: {len(prompts)}", flush=True)

    # Loaded one model at a time (each lens_trajectory frees before returning) to stay memory-safe.
    base_traj = lens_trajectory(C.load_base, prompts, n_layers)
    ft_traj = lens_trajectory(lambda: C.load_ft(merge=True), prompts, n_layers)

    lines = ["=" * 72, "LOGIT LENS  (per-layer last-token prediction: base vs fine-tuned)", "=" * 72]
    for i, (sample, bt, ft) in enumerate(zip(C.EVAL_SAMPLES, base_traj, ft_traj)):
        b_final, f_final = bt[-1], ft[-1]
        # "emergence" = first layer whose top token == the model's final top token.
        b_emg = next((L for L in range(n_layers) if bt[L] == b_final), n_layers - 1)
        f_emg = next((L for L in range(n_layers) if ft[L] == f_final), n_layers - 1)
        lines.append("")
        lines.append(f"PROMPT {i + 1}: {sample[:60]}...")
        lines.append(f"  final token   base={tok.decode([b_final])!r}  ft={tok.decode([f_final])!r}"
                     f"   ({'DIVERGE' if b_final != f_final else 'agree'})")
        lines.append(f"  emerges @layer base={b_emg}  ft={f_emg}")
        lines.append(f"  {'layer':>5} | {'base top':>16} | {'ft top':>16}")
        for L in range(n_layers):
            mark = "  <--" if (L == b_emg or L == f_emg) else ""
            lines.append(f"  {L:>5} | {tok.decode([bt[L]])!r:>16} | {tok.decode([ft[L]])!r:>16}{mark}")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(os.path.join(C.RESULTS, "logit_lens.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[saved] {os.path.join(C.RESULTS, 'logit_lens.txt')}")


if __name__ == "__main__":
    main()
