"""The two-ended contrast: the paper's main test.

Comparing one end of the entropy scale against a random control requires
deciding which end to report, and that decision can only be made after the
students are trained. The difference between the two ends requires no such
decision. It is symmetric in the two arms, so it carries no selection bias,
and it is the quantity that answers whether the score moves accuracy at all.

Reads data/per_run_accuracies.csv and nothing else. Averages the two seeds
inside a fold first, so one fold is one observation, then runs a two-sided
paired t-test on the five fold-level differences (df = 4).

    python code/16_contrast/two_ended_contrast.py
    python code/16_contrast/two_ended_contrast.py --arms gated_H H_high
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "per_run_accuracies.csv"

# label, the scores whose two ends are contrasted, and the group each end
# lives in. Most cells ran both ends in one group; on Qwen2-1.5B the low end
# was run later, in its own group, which is why that cell names two. The paper
# labels those arms post hoc for the same reason.
CELLS = [
    ("GSM8K / Qwen1.5-1.8B",     ["gated", "mean"], "fold_selection_q15_18"),
    ("GSM8K / Qwen2-1.5B",       ["gated"],         "fold_selection_q2_15",
                                                    "fold_selection_q2_15_both_ends"),
    ("MATH / Qwen2-1.5B",        ["gated", "mean"], "fold_selection_math2_q2_15"),
    ("ECQA / Qwen1.5-1.8B",      ["gated", "mean"], "fold_ecqa_q15_18"),
    ("Orca-Math / Qwen1.5-1.8B", ["gated"],         "fold_orca_anchor_half"),
    ("APE210K / Qwen1.5-1.8B",   ["gated", "mean"], "fold_ape210k_ends"),
]
ENDS = {"gated": ("gated_H", "gated_H_low"), "mean": ("H_high", "H_low")}


def t_sf(t, df):
    """Two-sided tail of Student's t, via the regularised incomplete beta."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2*m - 1) * (a + 2*m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2*m) * (a + 2*m + 1))
        d = 1.0 / (1e-30 if abs(1.0 + num * d) < 1e-30 else 1.0 + num * d)
        c = 1e-30 if abs(1.0 + num / c) < 1e-30 else 1.0 + num / c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def load():
    by = defaultdict(lambda: defaultdict(list))
    if not CSV.exists():
        sys.exit(f"missing {CSV}")
    with open(CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["rule"] and r["fold"]:
                by[r["group"]][(r["fold"], r["rule"])].append(
                    100 * float(r["acc_lenient"]))
    return by


def contrast(cells, hi, lo):
    folds = sorted({f for f, r in cells if r in (hi, lo)})
    diffs = []
    for f in folds:
        a, b = cells.get((f, hi)), cells.get((f, lo))
        if a and b:
            diffs.append(sum(a) / len(a) - sum(b) / len(b))
    n = len(diffs)
    if n < 2:
        return None
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t = m / se if se else float("inf")
    tcrit = 2.7764451051977987 if n == 5 else None
    return dict(mean=m, sd=sd, n=n, p=t_sf(abs(t), n - 1),
                floor=tcrit * se if tcrit else float("nan"),
                wins=sum(1 for d in diffs if d > 0), diffs=diffs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs=2, metavar=("HIGH", "LOW"),
                    help="contrast these two rule names instead of the defaults")
    ap.add_argument("--verbose", action="store_true",
                    help="print the five fold-level differences")
    args = ap.parse_args()
    by = load()

    print("High-scoring selection minus low-scoring selection, paired at "
          "the fold.\n")
    print(f"{'cell':<26} {'score':<6} {'high-low':>9} {'sd':>6} {'floor':>6} "
          f"{'p':>8} {'folds':>6}")
    results = []
    for label, scores, *groups in CELLS:
        # merge the groups this cell spans, so an arm run later still pairs
        cells = {}
        for g in groups:
            cells.update(by[g])
        for which in scores:
            hi, lo = ENDS[which] if not args.arms else args.arms
            r = contrast(cells, hi, lo)
            if r is None:
                continue
            results.append((label, which, r))
            print(f"{label:<26} {which:<6} {r['mean']:>+9.2f} {r['sd']:>6.2f} "
                  f"{r['floor']:>6.2f} {r['p']:>8.4f} {r['wins']:>3}/{r['n']}")
            if args.verbose:
                print(f"{'':<26} folds: "
                      + "  ".join(f"{d:+.2f}" for d in r["diffs"]))
            if args.arms:
                break

    resolved = [r for _, _, r in results if r["p"] < 0.05]
    alpha = 0.05 / len(results) if results else float("nan")
    survive = [(l, w, r) for l, w, r in results if r["p"] < alpha]
    print(f"\n{len(resolved)} of {len(results)} contrasts resolve at 0.05.")
    print(f"Treating all {len(results)} as one family, Bonferroni at "
          f"{alpha:.4f} leaves {len(survive)}:")
    for label, which, r in survive:
        print(f"    {r['mean']:+6.2f}  {label} ({which} entropy)")
    signs = {"+" if r["mean"] > 0 else "-" for _, _, r in survive}
    if len(signs) > 1:
        print("\nThe surviving contrasts do not share a sign.")


if __name__ == "__main__":
    main()
