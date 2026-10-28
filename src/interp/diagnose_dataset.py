"""DATASET DIAGNOSTIC (no GPU, no model needed) — is a dataset's 'plain' target actually plainer?

This is the empirical justification for the whole clean-dataset fix. For every dataset in the
registry it loads n training rows and measures the mean Flesch-Kincaid grade of the SOURCE
(technical) vs the TARGET ('plain') text. The sign of the delta is the whole story:

    delta > 0  ->  the 'plain' target is HARDER than the source  = the Goodhart trap
                   (maximising ROUGE against it teaches register-capture, not simplification)
    delta < 0  ->  the target really is simpler                  = a dataset worth fine-tuning on

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


def fk_grade(text):
    import textstat

    return textstat.flesch_kincaid_grade(text) if text and text.strip() else float("nan")


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
    verdict = ("TRAP — target is HARDER than source" if delta > 0.5 else
               "CLEAN — target is simpler than source" if delta < -0.5 else
               "NEUTRAL — target ~ source grade")
    lines.append(f"  n={len(ds)}  mean FK grade:  SOURCE {ms:5.1f}  ->  TARGET {mt:5.1f}   "
                 f"(delta {delta:+.1f})   => {verdict}")
    for ex in ds.select(range(min(show, len(ds)))):
        lines.append("-" * 90)
        lines.append(f"  SOURCE (g{fk_grade(ex['complex']):.0f}): {ex['complex'][:220]}")
        lines.append(f"  TARGET (g{fk_grade(ex['simple']):.0f}): {ex['simple'][:220]}")
    return lines, delta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="rows per dataset to average over")
    ap.add_argument("--show", type=int, default=2, help="example pairs to print per dataset")
    args = ap.parse_args()

    all_lines, summary = [], []
    for name in C.DATASETS:
        lines, delta = diagnose_one(name, args.n, args.show)
        all_lines += lines + [""]
        if delta is not None:
            summary.append((name, delta))

    all_lines.append("=" * 90)
    all_lines.append("SUMMARY (source->target FK-grade delta; negative = genuinely simpler):")
    for name, delta in sorted(summary, key=lambda x: x[1]):
        tag = "CLEAN" if delta < -0.5 else "TRAP" if delta > 0.5 else "neutral"
        all_lines.append(f"   {name:>8} : {delta:+.1f}   [{tag}]")
    all_lines.append("Fine-tune on a CLEAN dataset; keep the TRAP one as the failure case.")

    report = "\n".join(all_lines)
    print(report)
    os.makedirs(C.RESULTS, exist_ok=True)
    out = os.path.join(C.RESULTS, "dataset_diagnostic.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
