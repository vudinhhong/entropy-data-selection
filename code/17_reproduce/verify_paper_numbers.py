"""Recompute the paper's headline numbers from data/per_run_accuracies.csv alone.

Every paired test in the paper is a two-sided t-test on five fold-level
differences, with the two training seeds inside a fold averaged first. This
script does exactly that, reads nothing but the released CSV, and prints the
recomputed value beside the value printed in the paper so a reader can see
whether they agree.

    python code/17_reproduce/verify_paper_numbers.py
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "per_run_accuracies.csv"
TCRIT4 = 2.7764451051977987          # t_{0.975,4}


def student_t_sf(t, df):
    """Two-sided tail of Student's t, by the regularised incomplete beta."""
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m, num = i // 2, 0.0
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2*m - 1) * (a + 2*m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2*m) * (a + 2*m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def load():
    by = defaultdict(lambda: defaultdict(list))      # [group][fold,rule] -> accs
    with open(CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r["rule"] or not r["fold"]:
                continue
            by[r["group"]][(r["fold"], r["rule"])].append(100 * float(r["acc_lenient"]))
    return by


def paired(by, group, arm, ref):
    """Fold-level paired difference between two arms of one group."""
    cells = by[group]
    folds = sorted({f for f, r in cells if r in (arm, ref)})
    diffs = []
    for f in folds:
        a, b = cells.get((f, arm)), cells.get((f, ref))
        if a and b:
            diffs.append(sum(a)/len(a) - sum(b)/len(b))
    n = len(diffs)
    if n < 2:
        return None
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t = m / se if se else float("inf")
    return dict(mean=m, sd=sd, n=n, t=t, p=student_t_sf(abs(t), n - 1),
                floor=TCRIT4 * se, wins=sum(1 for d in diffs if d > 0))


# (label, group, arm, reference arm, value printed in the paper)
CHECKS = [
    ("Table 2  GSM8K gated H vs random",      "fold_selection_q15_18",   "gated_H",     "random", +1.33),
    ("Table 2  GSM8K mean H low vs random",   "fold_selection_q15_18",   "H_low",       "random", -1.40),
    ("Table 2  MATH gated H vs random",       "fold_selection_math2_q2_15",    "gated_H",     "random", +0.20),
    ("Table 2  ECQA gated H low vs random",   "fold_ecqa_q15_18",           "gated_H_low", "random", +1.81),
    ("Table 2  ECQA mean H high vs random",   "fold_ecqa_q15_18",           "H_high",      "random", -3.97),
    ("Table 4  Orca gated H vs random",       "fold_orca_anchor_half", "gated_H",   "random", -1.59),
    ("Table 4  Orca length vs random",        "fold_orca_anchor_half", "length",    "random", -1.15),
    ("Table 4  APE210K mean H high vs random","fold_ape210k_ends",   "H_high",      "random", -2.76),
    ("Contrast GSM8K mean H high - low",      "fold_selection_q15_18",   "H_high",      "H_low",  +2.35),
    ("Contrast GSM8K gated high - low",       "fold_selection_q15_18",   "gated_H",     "gated_H_low", +1.36),
    ("Contrast ECQA mean H high - low",       "fold_ecqa_q15_18",           "H_high",      "H_low",  -5.88),
    ("Contrast Orca gated high - low",        "fold_orca_anchor_half", "gated_H",   "gated_H_low", -2.47),
    ("Contrast APE210K mean high - low",      "fold_ape210k_ends",   "H_high",      "H_low",  -3.46),
    ("Length test GSM8K gated - mean H",      "fold_selection_q15_18",   "gated_H",     "H_high", +0.38),
    ("Table 10 untrained teacher vs random",  "fold_rawteacher_q15_18",      "gated_H", "random", +1.43),
    ("Table 10 untrained, whole split",       "fold_rawteacher_full_q15_18", "gated_H", "random", +1.01),
    ("Llama-3.2-3B teacher vs random",        "fold_llamateacher_q15_18",    "gated_H", "random", +1.89),
    ("ECQA option entropy, high end",         "fold_ecqa_q15_18",      "answer_H_high", "random", -2.78),
    ("Cross-family Llama at a tenth",         "fold_cross_llama_gsm8k",      "gated_H", "random", +0.06),
    ("Cross-family SmolLM2 at a tenth",       "fold_cross_smol_gsm8k",       "gated_H", "random", -0.05),
]


# Table 9 in full: (label, args, mean, median, worst, cells lost, decides %,
# wrong %). A fixed policy neither decides nor can be wrong, so it carries
# None for the last two.
PILOT = [
    ("always the high end",      ("fixed", "H"),      -0.16, +0.19, -3.97, 8, None, None),
    ("always the low end",       ("fixed", "L"),      +0.02, +0.11, -2.82, 11, None, None),
    ("oracle direction",         ("fixed", "oracle"), +0.82, +0.72, -0.04, 1, None, None),
    ("pilot, sign, m=2, 1 seed", ("rule", 2, 1, "sign"),
                                                      +0.74, +0.70, -0.16, 1, 100, 11),
    ("pilot, sign, m=3, 2 seeds", ("rule", 3, 2, "sign"),
                                                      +0.77, +0.70, -0.31, 2, 100, 6),
    ("pilot, interval, m=2",     ("rule", 2, 2, "interval"),
                                                      +0.06, +0.03, -0.06, 2, 8, 11),
    ("pilot, interval, m=4",     ("rule", 4, 2, "interval"),
                                                      +0.50, +0.14, -0.26, 2, 50, 0),
]
N_CELLS = 23


def check_pilot():
    """Recompute the direction-pilot table and compare with the paper."""
    import importlib.util
    path = CSV.parents[1] / "code" / "18_direction" / "direction_pilot.py"
    if not path.exists():
        print("\nskipping the pilot table: 18_direction is not in this archive")
        return 0
    spec = importlib.util.spec_from_file_location("direction_pilot", path)
    dp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dp)
    cells = dp.load("acc_lenient")
    print(f"\nTable 9, Pilot-Based Direction Selection ({len(cells)} cells)")
    bad = 0 if len(cells) == N_CELLS else 1
    if bad:
        print(f"  expected {N_CELLS} cells with both ends, found {len(cells)}")
    import statistics
    for label, args, *paper in PILOT:
        if args[0] == "fixed":
            vals = [v for _, v in dp.fixed(cells, args[1])]
            rates = [None, None]
        else:
            _, m, seeds, rule = args
            per, decides, wrong = dp.evaluate(cells, m, seeds, rule)
            vals = [v for _, v in per]
            rates = [round(100 * decides), round(100 * wrong)]
        got = [sum(vals) / len(vals), statistics.median(vals), min(vals),
               sum(1 for v in vals if v < 0)] + rates
        # tolerances: two places on the point values, exact on the counts,
        # one percentage point on the rates the table rounds.
        tol = [0.005, 0.005, 0.005, 0.5, 1.0, 1.0]
        off = [i for i, (g, p) in enumerate(zip(got, paper))
               if p is not None and abs(g - p) > tol[i]]
        bad += bool(off)
        cols = ("  ".join(f"{g:+.2f}" for g in got[:3])
                + f"  {got[3]}/{len(vals)}"
                + ("" if rates[0] is None
                   else f"  {rates[0]:>3d}%  {rates[1]:>2d}%"))
        print(f"  {label:<28} {cols:<42} "
              + ("ok" if not off else
                 "MISMATCH in " + ",".join(("mean", "median", "worst",
                                            "loses", "decides",
                                            "wrong")[i] for i in off)))
    return bad


def main():
    if not CSV.exists():
        sys.exit(f"missing {CSV}")
    by = load()
    print(f"{'what':<42} {'paper':>7} {'here':>7} {'sd':>6} {'p':>8} {'folds':>6}  ")
    bad = 0
    for label, group, arm, ref, paper in CHECKS:
        r = paired(by, group, arm, ref)
        if r is None:
            print(f"{label:<42} {paper:>+7.2f}   no data in the released CSV")
            bad += 1
            continue
        agree = abs(r["mean"] - paper) <= 0.015
        bad += not agree
        print(f"{label:<42} {paper:>+7.2f} {r['mean']:>+7.2f} {r['sd']:>6.2f} "
              f"{r['p']:>8.4f} {r['wins']:>3}/{r['n']}  {'ok' if agree else 'MISMATCH'}")
    print(f"\n{len(CHECKS) - bad} of {len(CHECKS)} reproduce from the CSV alone.")

    bad += check_pilot()

    # Detection floors, computed from the unrounded standard deviation and
    # printed to three places so the values quoted in the tables can be
    # checked at the precision they appear in.
    print("\nDetection floors, t(0.975,4) * s / sqrt(5), to three places:")
    for label, group, arm, ref, _ in CHECKS:
        r = paired(by, group, arm, ref)
        if r:
            print(f"  {label:<42} {r['floor']:.3f}  -> prints as {r['floor']:.2f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
