"""DATASET DIAGNOSTIC (no GPU, no model needed) — is a dataset's 'plain' target actually plainer?

This is the empirical justification for the whole clean-dataset fix. For every dataset in the
registry it loads n training rows and measures the mean Flesch-Kincaid grade of the SOURCE
(technical) vs the TARGET ('plain') text.

TWO conditions must BOTH hold for a dataset to be worth fine-tuning on — learned the hard way:
    (1) delta < 0            ->  the target is simpler than its source (relative), AND
    (2) abs target grade < PLAIN_GRADE_MAX  ->  the target is actually plain (absolute).

Condition (1) ALONE is a Goodharted proxy. 'scilay' passed it (source 17.1 -> target 16.4,
delta -0.7) yet fine-tuning on it made the model WORSE (grade 8.6 -> 14.2, 0 -> 13 fabrications):
a target that is 'simpler than a grade-17 abstract' is still grade-16 and nowhere near plain.
So a negative delta earns the label CLEAN only if the absolute target grade is also plain.

Run (anywhere, including your laptop — only needs `datasets` + `textstat`):
    python src/interp/diagnose_dataset.py                 # all registered datasets, n=200
    python src/interp/diagnose_dataset.py --n 500
Writes results/interp/dataset_diagnostic.txt
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_common as C  # noqa: E402

# Absolute FK grade a 'plain' target must come in UNDER to count as genuinely plain. Patient-facing
# health materials are written for ~grade 6-8; 10 is a generous ceiling. Above it, a dataset is a
# trap no matter how much simpler its targets are than their (even harder) sources.
PLAIN_GRADE_MAX = 10.0


def fk_grade(text):
    import textstat

    return textstat.flesch_kincaid_grade(text) if text and text.strip() else float("nan")


def classify(ms, mt):
    """Verdict from BOTH the relative delta AND the absolute target grade (tag, message).
    CLEAN requires target simpler than source AND absolutely plain; a sub-threshold source that
    the target still can't beat is the trap the relative-delta-only diagnostic used to miss."""
    delta = mt - ms
    if mt >= PLAIN_GRADE_MAX:
        rel = ("simpler than source but still" if delta < -0.5 else
               "and harder than source;" if delta > 0.5 else "and ~ source grade;")
        return "TRAP", f"TRAP — target grade {mt:.1f} >= {PLAIN_GRADE_MAX:.0f} ({rel} not plain)"
    if delta < -0.5:
        return "CLEAN", f"CLEAN — target simpler AND plain (grade {mt:.1f} < {PLAIN_GRADE_MAX:.0f})"
    if delta > 0.5:
        return "TRAP", "TRAP — target is HARDER than source"
    return "NEUTRAL", "NEUTRAL — target ~ source grade"


def diagnose_one(name, n, show):
    lines = ["=" * 90, f"DATASET {name!r}  ({C.DATASETS[name]['hf']})",
             f"  {C.DATASETS[name]['note']}"]
    try:
        ds = C.load_pairs(name=name, split="train", n=n)
    except Exception as e:  # a dataset may be renamed/gated; report, don't crash the whole run
        lines.append(f"  [skipped] could not load: {type(e).__name__}: {str(e)[:120]}")
        return lines, None

    src = [g for g in (fk_grade(ex["complex"]) for ex in ds) if g == g]  # drop NaNs
    tgt = [g for g in (fk_grade(ex["simple"]) for ex in ds) if g == g]
    ms, mt = statistics.mean(src), statistics.mean(tgt)
    delta = mt - ms
    tag, verdict = classify(ms, mt)
    lines.append(f"  n={len(ds)}  mean FK grade:  SOURCE {ms:5.1f}  ->  TARGET {mt:5.1f}   "
                 f"(delta {delta:+.1f}, abs target {mt:.1f})   => {verdict}")
    for ex in ds.select(range(min(show, len(ds)))):
        lines.append("-" * 90)
        lines.append(f"  SOURCE (g{fk_grade(ex['complex']):.0f}): {ex['complex'][:220]}")
        lines.append(f"  TARGET (g{fk_grade(ex['simple']):.0f}): {ex['simple'][:220]}")
    return lines, (delta, mt, tag)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="rows per dataset to average over")
    ap.add_argument("--show", type=int, default=2, help="example pairs to print per dataset")
    args = ap.parse_args()

    all_lines, summary = [], []
    for name in C.DATASETS:
        lines, info = diagnose_one(name, args.n, args.show)
        all_lines += lines + [""]
        if info is not None:
            summary.append((name, *info))   # (name, delta, abs_target_grade, tag)

    all_lines.append("=" * 90)
    all_lines.append(
        f"SUMMARY (CLEAN needs BOTH: delta < 0 AND abs target grade < {PLAIN_GRADE_MAX:.0f}):")
    for name, delta, mt, tag in sorted(summary, key=lambda x: x[1]):
        all_lines.append(f"   {name:>8} : delta {delta:+.1f}   abs-target g{mt:4.1f}   [{tag}]")
    all_lines.append("Fine-tune on a CLEAN dataset; keep the TRAP ones as failure cases.")

    report = "\n".join(all_lines)
    print(report)
    os.makedirs(C.RESULTS, exist_ok=True)
    out = os.path.join(C.RESULTS, "dataset_diagnostic.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
