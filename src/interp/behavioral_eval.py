"""
Behavioral check (NOT interpretability yet): does the fine-tune actually simplify, or did it
learn the dataset's review-summary register and start hallucinating?

For each probe passage it generates from BOTH models (loaded one at a time to stay memory-safe),
then reports three things per output:
  - Flesch-Kincaid grade  (readability; lower = simpler)  -- what ROUGE can't see
  - fabricated numbers     (numbers in the output absent from the input = invented specifics)
  - the text itself        (so you can read what changed)

It also runs a dataset diagnostic: the mean FK grade of the dataset's own 'plain' answers, to
test whether the targets themselves are high-grade Cochrane-style summaries (which would explain
why matching them raised both ROUGE and reading grade).

Run:  python src/interp/behavioral_eval.py           # uses the persisted adapter
Writes a copy of the report to results/interp/behavioral_eval.txt
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_common as C  # noqa: E402


def fk_grade(text):
    import textstat

    return textstat.flesch_kincaid_grade(text) if text.strip() else float("nan")


def generate_both(samples, max_new_tokens):
    """Base pass, free it, then ft pass — never both resident at once."""
    base = C.load_base()
    base_outs = [C.generate(base, t, max_new_tokens) for t in samples]
    C.free(base)

    ft = C.load_ft(merge=True)
    ft_outs = [C.generate(ft, t, max_new_tokens) for t in samples]
    C.free(ft)
    return base_outs, ft_outs


def dataset_diagnostic(lines, n=100, show=3):
    ds = C.load_pairs(split="train", n=n)
    q = statistics.mean(fk_grade(ex["complex"]) for ex in ds)
    a = statistics.mean(fk_grade(ex["simple"]) for ex in ds)
    lines.append("\n" + "=" * 90)
    lines.append(f"DATASET {C.DATASET!r} (n={len(ds)})  mean FK grade  |  technical INPUT {q:.1f}  ->  "
                 f"reference 'plain' ANSWER {a:.1f}  (delta {a - q:+.1f})")
    lines.append("If the 'plain' answers are themselves ~grade 11-12 review summaries, matching "
                 "them raises\nboth ROUGE and reading grade — the fine-tune learned the register, "
                 "not 'make it simpler'. A NEGATIVE delta means the targets really are simpler.")
    for ex in ds.select(range(show)):
        lines.append("-" * 90)
        lines.append(f"  SOURCE: {ex['complex'][:260]}")
        lines.append(f"  TARGET: {ex['simple'][:260]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--diagnostic-n", type=int, default=100)
    ap.add_argument("--no-diagnostic", action="store_true")
    args = ap.parse_args()

    print("device:", C.DEVICE, "| adapter:", C.ADAPTER_DIR, flush=True)
    base_outs, ft_outs = generate_both(C.EVAL_SAMPLES, args.max_new_tokens)

    lines = []
    base_grades, ft_grades = [], []
    for tech, b, f in zip(C.EVAL_SAMPLES, base_outs, ft_outs):
        bg, fg = fk_grade(b), fk_grade(f)
        base_grades.append(bg)
        ft_grades.append(fg)
        b_fab, f_fab = C.fabricated_numbers(tech, b), C.fabricated_numbers(tech, f)
        lines.append("=" * 90)
        lines.append(f"MEDICAL: {tech}")
        lines.append(f"\nBASE  (grade {bg:.1f} | fabricated #s: {len(b_fab)} {b_fab}):\n  {b}")
        lines.append(f"\nFT    (grade {fg:.1f} | fabricated #s: {len(f_fab)} {f_fab}):\n  {f}")

    lines.append("\n" + "=" * 90)
    lines.append(
        f"mean FK grade        |  base {statistics.mean(base_grades):.1f}  ->  "
        f"fine-tuned {statistics.mean(ft_grades):.1f}  "
        f"(delta {statistics.mean(ft_grades) - statistics.mean(base_grades):+.1f})"
    )
    tot_b = sum(len(C.fabricated_numbers(t, o)) for t, o in zip(C.EVAL_SAMPLES, base_outs))
    tot_f = sum(len(C.fabricated_numbers(t, o)) for t, o in zip(C.EVAL_SAMPLES, ft_outs))
    lines.append(f"total fabricated #s  |  base {tot_b}  ->  fine-tuned {tot_f}   "
                 "(higher = more invented specifics = worse faithfulness)")

    if not args.no_diagnostic:
        dataset_diagnostic(lines, n=args.diagnostic_n)

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(C.RESULTS, exist_ok=True)
    out_path = C.result_path("behavioral_eval")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
