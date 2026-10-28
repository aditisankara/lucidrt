"""THE SCOREBOARD — base vs fine-tuned on one dataset, judged by metrics ROUGE can't see.

The inference project reported a +37% ROUGE-L gain and stopped there. This script refuses to let
ROUGE be the headline. It generates from BOTH models on the eval split and reports THREE numbers:

  - ROUGE-L          the gamed proxy (word-overlap with the reference) -- shown, but NOT trusted
  - FK grade         readability; LOWER = simpler  -- the thing ROUGE is blind to
  - fabricated #s    numbers in the output absent from the source = invented specifics (worse)

On the TRAP dataset ('med') the story is: ROUGE up, FK grade up (worse), fabrications up (worse)
-- a Goodhart win. On a CLEAN dataset ('scilay') a real improvement looks like: ROUGE up AND FK
grade DOWN AND fabrications flat/low. The scoreboard makes that distinction impossible to miss.

Run (GPU node; the LUCIDRT_DATASET adapter must already be trained):
    python src/interp/scoreboard.py                 # n=40 eval rows
    LUCIDRT_DATASET=scilay python src/interp/scoreboard.py
Writes results/interp/scoreboard[__<dataset>].txt
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


def _eval(model, rows, max_new_tokens):
    """Generate for every row; return (predictions, mean FK grade, total fabricated numbers)."""
    preds, grades, fab = [], [], 0
    for ex in rows:
        out = C.generate(model, ex["complex"], max_new_tokens)
        preds.append(out)
        g = fk_grade(out)
        if g == g:  # not NaN
            grades.append(g)
        fab += len(C.fabricated_numbers(ex["complex"], out))
    return preds, (statistics.mean(grades) if grades else float("nan")), fab


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40, help="eval rows (keep fixed across comparisons)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    if C.DEVICE != "cuda":
        sys.exit("Need a GPU node (generation).")

    spec = C.DATASETS[C.DATASET]
    rows = C.load_pairs(split=spec["eval_split"], n=args.n)
    refs = [ex["simple"] for ex in rows]
    ref_grade = statistics.mean(g for g in (fk_grade(r) for r in refs) if g == g)
    print(f"dataset: {C.DATASET} ({spec['hf']}) | eval rows: {len(rows)} | adapter: {C.ADAPTER_DIR}",
          flush=True)

    # one model resident at a time (12 GB-safe), exactly as behavioral_eval does
    base = C.load_base()
    base_preds, base_grade, base_fab = _eval(base, rows, args.max_new_tokens)
    C.free(base)

    ft = C.load_ft(merge=True)
    ft_preds, ft_grade, ft_fab = _eval(ft, rows, args.max_new_tokens)
    C.free(ft)

    import evaluate

    rouge = evaluate.load("rouge")
    base_r = rouge.compute(predictions=base_preds, references=refs)["rougeL"]
    ft_r = rouge.compute(predictions=ft_preds, references=refs)["rougeL"]
    pct = (ft_r - base_r) / base_r * 100 if base_r else float("nan")

    def arrow(base_v, ft_v, lower_is_better):
        d = ft_v - base_v
        if abs(d) < 1e-9:
            return "flat"
        good = (d < 0) if lower_is_better else (d > 0)
        return ("better" if good else "WORSE") + f" ({d:+.2f})"

    L = ["=" * 78,
         f"SCOREBOARD — base vs fine-tuned   |   dataset={C.DATASET!r}  n={len(rows)}  greedy",
         "=" * 78,
         f"{'metric':<34}{'base':>10}{'fine-tuned':>13}   verdict",
         "-" * 78,
         f"{'ROUGE-L  (GAMED PROXY — distrust)':<34}{base_r:>10.3f}{ft_r:>13.3f}   "
         f"{pct:+.0f}%  (not evidence of quality)",
         f"{'FK grade  (lower = simpler)':<34}{base_grade:>10.1f}{ft_grade:>13.1f}   "
         f"{arrow(base_grade, ft_grade, lower_is_better=True)}",
         f"{'fabricated #s (total; lower better)':<34}{base_fab:>10d}{ft_fab:>13d}   "
         f"{arrow(base_fab, ft_fab, lower_is_better=True)}",
         "-" * 78,
         f"reference-target mean FK grade: {ref_grade:.1f}  "
         f"({'targets are HARDER than typical speech — the trap' if ref_grade > 10 else 'targets are genuinely plain'})",
         "",
         "Reading this: ROUGE going UP means only 'looks more like the reference'. A GENUINE",
         "improvement is ROUGE up AND FK grade DOWN AND fabricated #s flat/low. If ROUGE rises",
         "while FK grade also rises, the gain is Goodhart register-capture, not simplification."]

    report = "\n".join(L)
    print("\n" + report)
    os.makedirs(C.RESULTS, exist_ok=True)
    out = C.result_path("scoreboard")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
