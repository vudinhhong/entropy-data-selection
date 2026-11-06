#!/usr/bin/env python3
"""Retrospective evaluation of Pilot-Based Direction Selection, PDS,
which is Algorithm 1 of the paper.

Every cell in the released table that ran both ends of the entropy scale and
a random control gives five fold-level values per arm. A pilot that sees m of
those folds, decides a direction from them and is then judged on the folds it
did not see is exactly a leave-folds-out evaluation of the decision rule, and
the table has enough cells to run it on all of them.

Two rules are compared. The sign rule takes the direction of the pilot's mean
contrast. The interval rule takes a direction only when a paired t interval on
the pilot folds excludes zero, and otherwise selects at random. Four fixed
policies bound the result: always high, always low, random, and an oracle that
is told the direction the full five folds imply.

    python3 direction_pilot.py            # the table the paper reports
    python3 direction_pilot.py --strict   # strict accuracy instead of lenient

Stdlib only, no third-party dependency.
"""
import argparse, csv, itertools, math, statistics as st
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "data" / "per_run_accuracies.csv"

# Two-sided 0.05 critical values of Student's t, df = 1..4.
TCRIT = {1: 12.7062047361747, 2: 4.302652729911275,
         3: 3.182446305284263, 4: 2.776445105197796}

# A cell is one (corpus, student, teacher, pool, budget) configuration that ran
# both ends of a score family and a stratified random control over the same
# five folds. They are discovered from the table rather than listed, so a cell
# cannot be left out by hand. One cell spans two groups, because its low end
# was run in a later batch; the pair is declared here and nowhere else.
MERGED = {"fold_selection_q2_15": "fold_selection_q2_15_both_ends"}

# Printable names, keyed by group. A group absent from here prints its own name.
NAMES = {
    "fold_selection_q15_18":       "GSM8K / Qwen1.5-1.8B",
    "fold50_anchor_q15_18":        "GSM8K / Qwen1.5-1.8B b374",
    "fold_llamateacher_q15_18":    "GSM8K / Qwen1.5-1.8B, Llama teacher",
    "fold_rawteacher_q15_18":      "GSM8K / Qwen1.5-1.8B, untrained teacher",
    "fold_rawteacher_full_q15_18": "GSM8K / Qwen1.5-1.8B, untrained, full pool",
    "fold_selection_q2_15":        "GSM8K / Qwen2-1.5B",
    "fold50_llama_gsm8k":          "GSM8K / Llama-3.2-1B b374",
    "fold_cross_llama_gsm8k":      "GSM8K / Llama-3.2-1B b75",
    "fold50_smol_gsm8k":           "GSM8K / SmolLM2-360M b374",
    "fold_cross_smol_gsm8k":       "GSM8K / SmolLM2-360M b75",
    "fold_ecqa_q15_18":            "ECQA / Qwen1.5-1.8B",
    "fold_selection_math2_q2_15":  "MATH / Qwen2-1.5B",
    "fold_ape210k_ends":           "APE210K / Qwen1.5-1.8B",
    "fold_orca_anchor_half":       "Orca-Math / Qwen1.5-1.8B",
}
ENDS = [("gated", "gated_H", "gated_H_low"), ("mean", "H_high", "H_low")]

# Configurations that were run at two budgets, 75 and 374 sequences per fold.
BUDGET_PAIRS = [
    ("GSM8K / Qwen1.5-1.8B [gated]",     "GSM8K / Qwen1.5-1.8B b374 [gated]"),
    ("GSM8K / Llama-3.2-1B b75 [gated]", "GSM8K / Llama-3.2-1B b374 [gated]"),
    ("GSM8K / Llama-3.2-1B b75 [mean]",  "GSM8K / Llama-3.2-1B b374 [mean]"),
    ("GSM8K / SmolLM2-360M b75 [gated]", "GSM8K / SmolLM2-360M b374 [gated]"),
    ("GSM8K / SmolLM2-360M b75 [mean]",  "GSM8K / SmolLM2-360M b374 [mean]"),
]


def load(metric):
    runs = defaultdict(list)          # (group, rule, fold) -> [run accuracies]
    with open(CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "reported" or not r["fold"] or not r["rule"]:
                continue
            runs[(r["group"], r["rule"], r["fold"])].append(100 * float(r[metric]))
    groups = sorted({g for g, _, _ in runs})
    cells = []
    for group in groups:
        if group in MERGED.values():
            continue                  # reached through the group it is merged into
        sources = [group] + ([MERGED[group]] if group in MERGED else [])
        for family, hi, lo in ENDS:
            arms = {}
            for name, rule in (("H", hi), ("L", lo), ("R", "random")):
                arms[name] = {f: runs[(g, rule, f)]
                              for g in sources for f in "01234"
                              if (g, rule, f) in runs}
            folds = sorted(set(arms["H"]) & set(arms["L"]) & set(arms["R"]))
            if len(folds) == 5:
                label = NAMES.get(group, group)
                cells.append((f"{label} [{family}]", arms, folds))
    return cells


def mean(xs):
    return sum(xs) / len(xs)


def pilot_contrast(arms, pilot, seeds):
    """Fold-level contrasts seen by a pilot that trains `seeds` runs per arm."""
    return [mean(arms["H"][f][:seeds]) - mean(arms["L"][f][:seeds])
            for f in pilot]


def evaluate(cells, m, seeds, rule, fallback=None):
    """Mean held-out gain over every choice of m pilot folds, per cell.

    `fallback` is what the interval rule does when it does not resolve: None
    selects at random, "H" falls back to the default of taking the high end,
    which is the comparison that does not hand the sign rule the cases the
    interval abstains from.
    """
    per_cell, errors, trials, fired = [], 0, 0, 0
    for label, arms, folds in cells:
        truth = "H" if mean([mean(arms["H"][f]) - mean(arms["L"][f])
                             for f in folds]) > 0 else "L"
        gains = []
        for pilot in itertools.combinations(folds, m):
            held = [f for f in folds if f not in pilot]
            d = pilot_contrast(arms, pilot, seeds)
            if rule == "sign":
                choice = "H" if mean(d) > 0 else "L"
            else:
                half = TCRIT[len(d) - 1] * st.stdev(d) / math.sqrt(len(d))
                choice = ("H" if mean(d) - half > 0 else
                          "L" if mean(d) + half < 0 else None)
            trials += 1
            if choice:
                fired += 1
            else:
                choice = fallback
            if choice and choice != truth:
                errors += 1
            gains.append(0.0 if choice is None else
                         mean([mean(arms[choice][f]) - mean(arms["R"][f])
                               for f in held]))
        per_cell.append((label, mean(gains)))
    # Without a fallback the error rate is taken over the splits that
    # returned a direction; with one every split returns a direction.
    decided = trials if fallback else max(fired, 1)
    return per_cell, fired / trials, errors / decided


def escalating(cells, eps, m_start=2, m_max=4, seeds=1):
    """The sizing step of PDS: add folds while the bound exceeds eps.

    Escalation needs an order in which the folds arrive, so every permutation
    of the five is run and the result averaged.
    """
    per_cell, spent = [], []
    for _, arms, folds in cells:
        gains = []
        for order in itertools.permutations(folds):
            m = m_start
            while True:
                pilot = order[:m]
                d = pilot_contrast(arms, pilot, seeds)
                if 0.17 * st.stdev(d) / math.sqrt(m) <= eps or m >= m_max:
                    break
                m += 1
            held = [f for f in folds if f not in pilot]
            choice = "H" if mean(d) > 0 else "L"
            gains.append(mean([mean(arms[choice][f]) - mean(arms["R"][f])
                               for f in held]))
            spent.append(m)
        per_cell.append(mean(gains))
    return per_cell, mean(spent)


def fixed(cells, policy):
    out = []
    for label, arms, folds in cells:
        if policy == "oracle":
            pick = "H" if mean([mean(arms["H"][f]) - mean(arms["L"][f])
                                for f in folds]) > 0 else "L"
        else:
            pick = policy
        out.append((label, mean([mean(arms[pick][f]) - mean(arms["R"][f])
                                 for f in folds])))
    return out


def spread(cells, seeds=1):
    """Fold-level sd of the contrast, the quantity the regret bound uses.

    The bound has to be evaluated at the number of seeds the pilot trains:
    averaging two runs per arm before differencing shrinks the spread by
    about a third, and a pilot that trains one seed never sees that.
    """
    return [st.stdev([mean(a["H"][f][:seeds]) - mean(a["L"][f][:seeds])
                      for f in fs])
            for _, a, fs in cells]


def regret(cells, m, seeds):
    """Per-cell regret against the oracle, beside that cell's own bound."""
    pilot = dict(evaluate(cells, m, seeds, "sign")[0])
    best = dict(fixed(cells, "oracle"))
    sd = dict(zip((label for label, _, _ in cells), spread(cells, seeds)))
    return [(label, best[label] - pilot[label],
             0.17 * sd[label] / math.sqrt(m)) for label in pilot]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="score with strict accuracy instead of lenient")
    args = ap.parse_args()
    cells = load("acc_strict" if args.strict else "acc_lenient")
    print(f"{len(cells)} cells with both ends and a random control\n")

    print("Fixed policies, all five folds, mean accuracy against the random "
          "control (points)")
    for policy, name in (("H", "always the high end"),
                         ("L", "always the low end"),
                         ("oracle", "oracle direction")):
        r = fixed(cells, policy)
        print(f"  {name:22s} {mean([v for _, v in r]):+6.2f}   "
              f"worst cell {min(v for _, v in r):+6.2f}")
    print(f"  {'no selection (random)':22s} {0.0:+6.2f}   worst cell {0.0:+6.2f}")

    print("\nPilot rules, judged only on the folds the pilot did not see")
    print(f"  {'runs':>5} {'m':>2} {'seeds':>6} {'rule':>9} {'decides':>8} "
          f"{'wrong':>6} {'gain':>7} {'worst':>7}")
    for m in (2, 3, 4):
        for seeds in (1, 2):
            for rule in ("sign", "interval"):
                per, fire, err = evaluate(cells, m, seeds, rule)
                g = [v for _, v in per]
                print(f"  {2*m*seeds:>5} {m:>2} {seeds:>6} {rule:>9} "
                      f"{fire:>7.0%} {err:>5.0%} {mean(g):>+7.2f} "
                      f"{min(g):>+7.2f}")

    for seeds in (1, 2):
        sd = spread(cells, seeds)
        print(f"\nFold-level sd of the contrast at {seeds} seed per arm: "
              f"median {st.median(sd):.2f}, range {min(sd):.2f} to "
              f"{max(sd):.2f} points")
        print("Worst-case expected regret of the sign rule, 0.17 sd / sqrt(m):")
        for m in (2, 3, 4):
            print(f"  m = {m}: {0.17 * st.median(sd) / math.sqrt(m):.2f} points "
                  f"at the median sd, {0.17 * max(sd) / math.sqrt(m):.2f} "
                  f"at the largest")

    # The bound is an expectation under a normal model; the replay is one
    # realisation under a design that fixes each cell's truth at its five-fold
    # mean. Printing both lets a reader see that it is not a test of the bound.
    rows = regret(cells, 2, 1)
    over = [r for r in rows if r[1] > r[2]]
    print(f"\nPer-cell regret against the oracle, sign rule at m = 2, 1 seed")
    print(f"  mean {mean([r[1] for r in rows]):.2f}, "
          f"largest {max(r[1] for r in rows):.2f} "
          f"({max(rows, key=lambda r: r[1])[0]})")
    print(f"  cells whose regret exceeds their own bound: {len(over)}/{len(rows)}")

    print("\nSizing the pilot by the bound: add a fold while "
          "0.17 sd/sqrt(m) > eps")
    print(f"  {'eps':>5} {'mean m':>7} {'runs':>6} {'gain':>7} {'worst':>7}")
    for eps in (0.30, 0.20, 0.15, 0.10):
        per, m_bar = escalating(cells, eps)
        print(f"  {eps:>5.2f} {m_bar:>7.2f} {2*m_bar:>6.1f} "
              f"{mean(per):>+7.2f} {min(per):>+7.2f}")

    print("\nThe interval rule when it falls back to the default, not to random")
    print(f"  {'rule':24s} {'decides':>8} {'wrong':>6} {'mean':>7} {'worst':>7}")
    for m, seeds in ((2, 2), (4, 2)):
        for fb in (None, "H"):
            per, fire, err = evaluate(cells, m, seeds, "interval", fb)
            g = [v for _, v in per]
            name = f"m={m}, " + ("random" if fb is None else "default")
            print(f"  {name:24s} {fire:>7.0%} {err:>5.0%} "
                  f"{mean(g):>+7.2f} {min(g):>+7.2f}")

    print("\nWhat the pilot adds to the default, by where the cells sit")
    pilot = dict(evaluate(cells, 2, 1, "sign")[0])
    high, orac = dict(fixed(cells, "H")), dict(fixed(cells, "oracle"))
    keys = list(pilot)
    groups = [("all cells", keys),
              ("GSM8K cells", [k for k in keys if k.startswith("GSM8K")]),
              ("the other corpora", [k for k in keys if not k.startswith("GSM8K")])]
    by_corpus = defaultdict(list)
    for k in keys:
        by_corpus[k.split("/")[0].strip()].append(k)
    groups.append(("each corpus one vote", None))
    print(f"  {'':22s} {'n':>3} {'always H':>9} {'pilot':>7} {'adds':>7} {'oracle':>7}")
    for name, ks in groups:
        if ks is None:
            agg = lambda d: mean([mean([d[k] for k in v])
                                  for _, v in sorted(by_corpus.items())])
            n = len(by_corpus)
        else:
            agg = lambda d, ks=ks: mean([d[k] for k in ks])
            n = len(ks)
        print(f"  {name:22s} {n:>3} {agg(high):>+9.2f} {agg(pilot):>+7.2f} "
              f"{agg(pilot) - agg(high):>+7.2f} {agg(orac):>+7.2f}")

    print("\nThe same configuration at two budgets: does the direction hold?")
    print(f"  {'configuration':30s} {'b = 75':>16s} {'b = 374':>16s}  same sign")
    index = {label: (arms, folds) for label, arms, folds in cells}
    for small, large in BUDGET_PAIRS:
        if small not in index or large not in index:
            continue
        out = []
        for label in (small, large):
            arms, folds = index[label]
            d = [mean(arms["H"][f]) - mean(arms["L"][f]) for f in folds]
            m = mean(d)
            floor = TCRIT[4] * st.stdev(d) / math.sqrt(5)
            out.append((m, floor, abs(m) > floor))
        agree = (out[0][0] > 0) == (out[1][0] > 0)
        name = small.replace(" b75", "")
        print(f"  {name:30s} "
              f"{out[0][0]:>+7.2f} +-{out[0][1]:4.2f}{'*' if out[0][2] else ' '} "
              f"{out[1][0]:>+7.2f} +-{out[1][1]:4.2f}{'*' if out[1][2] else ' '}  "
              f"{'yes' if agree else 'no'}")
    print("  * exceeds its own detection floor, t(0.975,4) s/sqrt(5)")

    print("\nPer-corpus mean, sign rule at m = 2, one seed per arm")
    per, _, _ = evaluate(cells, 2, 1, "sign")
    always, oracle = dict(fixed(cells, "H")), dict(fixed(cells, "oracle"))
    by_corpus = defaultdict(list)
    for label, v in per:
        by_corpus[label.split("/")[0].strip()].append(
            (v, always[label], oracle[label]))
    print(f"  {'corpus':10s} {'cells':>5} {'pilot':>7} {'always H':>9} {'oracle':>7}")
    for corpus in sorted(by_corpus):
        rows = by_corpus[corpus]
        print(f"  {corpus:10s} {len(rows):>5} "
              f"{mean([a for a, _, _ in rows]):>+7.2f} "
              f"{mean([b for _, b, _ in rows]):>+9.2f} "
              f"{mean([o for _, _, o in rows]):>+7.2f}")

    print("\nPer-cell held-out gain, sign rule at m = 2, one seed per arm")
    per, _, _ = evaluate(cells, 2, 1, "sign")
    always = dict(fixed(cells, "H"))
    oracle = dict(fixed(cells, "oracle"))
    print(f"  {'cell':30s} {'pilot':>7} {'always H':>9} {'oracle':>7}")
    for label, v in per:
        print(f"  {label:30s} {v:>+7.2f} {always[label]:>+9.2f} {oracle[label]:>+7.2f}")


if __name__ == "__main__":
    main()
