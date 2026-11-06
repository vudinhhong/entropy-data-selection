"""Fig 1 of the paper, the selection result.

Panels A and B are computed from the per-run accuracies in
`data/per_run_accuracies.csv`, not from the tables, so the figure and Table
"Five selection rules on three corpora" are two renderings of one set of
numbers and cannot drift apart. Panel C is the truncation check, whose four
rows are small enough to carry here as literals.

PLOS wants vector or 300+ dpi raster at the final print width; we write PDF.

    python make_figures.py            # Fig 1
    python make_figures.py --budget   # also the budget curve, which the paper
                                      # reports in the text and does not plot

Figs 2 to 4 are drawn by make_pilot_figures.py in this directory.
"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Write beside the manuscript when there is one (the working tree), and into
# figures/ in the released archive, which carries no manuscript.
OUT = ROOT / "paper" if (ROOT / "paper").is_dir() else ROOT / "figures"
# Raw run directories, used only inside a working tree that still has them.
# The released archive ships the per-run table instead; see _load_runs.
GRIDS = [ROOT / "results" / "grid_v3", ROOT / "results" / "grid_v3" / "grid",
         ROOT / "results" / "grid"]

# Orca-Math is not in the selection table: it ran at its own budget and is the
# corpus the disagreement ordering was asked to predict. Its gated high end
# against the random control is -1.59 at a disagreement rate of 19.5%, which
# is below GSM8K's 25.3%, so the ordering predicts the wrong sign there.
ORCA_OOS = (19.5, -1.59)

# The three corpora of the selection table, in the order the paper uses: by how
# often the teacher's own top prediction differs from the reference text at the
# gated positions. That quantity is measured before any student is trained, but
# it was adopted as the ordering axis after the ECQA reversal had been seen, so
# panel A describes these three corpora and does not predict a fourth, see the
# out-of-sample test in the paper, where Orca-Math sits at 19.5% and reverses.
CORPORA = [("GSM8K", "fold_gsm8k_q15_18", 25.3),
           ("MATH", "fold_math2_q2_15", 44.5),
           ("ECQA", "fold_ecqa", 66.0)]

# Rule key -> label and marker. Only the five scored rules; `random` is the
# control every rule is paired against.
RULES = [("gated_H", "gated $H$, highest", "o", "#1f4e79"),
         ("H_high", "mean $H$, highest", "^", "#2e75b6"),
         ("nll_high", "perplexity, highest", "v", "#7f7f7f"),
         ("gated_H_low", "gated $H$, lowest", "s", "#b45f06"),
         ("H_low", "mean $H$, lowest", "D", "#e08214")]

# Table "Truncation": K, mean entropy error in nats, Spearman against the
# full-distribution ranking, overlap of the selected tenth.
TRUNCATION = [(5, 0.0134, 0.9960, 93.6), (10, 0.0034, 0.9995, 97.1),
              (20, 0.0010, 1.0000, 98.7), (50, 0.0003, 1.0000, 100.0)]

# Table "Budget": percent of the corpus, sequences, text-only, artifact.
BUDGET = [(1, 37, 34.67, 35.41), (2, 75, 33.59, 36.47), (5, 187, 34.12, 35.96),
          (10, 374, 33.36, 36.69), (20, 747, 34.39, 36.92)]
WHOLE_CORPUS, BASE = 34.27, 24.49


CSV = ROOT / "data" / "per_run_accuracies.csv"
_RUNS = None


def _load_runs():
    """Accuracies by run tag, from the released table or the raw run files.

    The released CSV is the normal source; the raw directories are used when
    this is run inside a working tree that still has them. Either way the
    numbers are identical, the CSV is built from those files by
    code/17_reproduce/build_run_table.py.
    """
    global _RUNS
    if _RUNS is not None:
        return _RUNS
    _RUNS = {}
    if CSV.exists():
        import csv as _csv
        with open(CSV, newline="", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                _RUNS[r["tag"]] = float(r["acc_lenient"]) * 100
    for d in GRIDS:
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            try:
                j = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if "tag" in j and "acc" in j:
                _RUNS.setdefault(j["tag"], j["acc"] * 100)
    if not _RUNS:
        raise SystemExit(
            f"no run data found. Expected {CSV} (shipped with the release) "
            f"or one of: " + ", ".join(str(d) for d in GRIDS))
    return _RUNS


def accuracy(tag):
    return _load_runs().get(tag)


def paired_runs(spec_name):
    """Per-run differences against the control, keyed by rule.

    One run is one (fold, training seed) pair. The rule and the control share
    the fold and the seed, so the difference is paired at the level the win
    count is read at. Fold means average the seeds first, which is the unit the
    t-test uses.
    """
    config = json.loads((ROOT / "code/05_results/config.json").read_text())
    spec = config["folds"][spec_name]
    control, seeds, n_folds = spec.get("control", "random"), spec["seeds"], spec["folds"]

    def runs(score):
        out = {}
        for f in range(n_folds):
            for seed in seeds:
                a = accuracy(spec["tag"].format(fold=f, score=score, seed=seed))
                if a is not None:
                    out[(f, seed)] = a
        return out

    base = runs(control)
    table = {}
    for score, *_ in RULES:
        if score not in spec["scores"]:
            continue
        arm = runs(score)
        keys = sorted(set(arm) & set(base))
        if not keys:
            continue
        per_run = [arm[k] - base[k] for k in keys]
        folds = sorted({f for f, _ in keys})
        per_fold = [st.mean(arm[k] for k in keys if k[0] == f)
                    - st.mean(base[k] for k in keys if k[0] == f) for f in folds]
        table[score] = {"runs": per_run, "folds": per_fold,
                        "wins": sum(1 for d in per_run if d > 0)}
    return table


def fig1(data):
    drawn = sum(len(cell["runs"]) for corpus in data.values()
                for cell in corpus.values())
    if drawn == 0:
        raise SystemExit(
            "fig1 would be empty: no runs matched the tags in "
            f"code/05_results/config.json. Refusing to overwrite {OUT/'fig1.pdf'}.")

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 300,
                         # U+2212 can fail to render or extract in some PDF
                         # pipelines, which silently drops the minus sign on a
                         # negative tick; the ASCII hyphen always survives.
                         "axes.unicode_minus": False,
                         # TrueType rather than the Type 3 default, so the
                         # figure converts to EPS with embedded fonts a
                         # journal's preflight check accepts.
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.5, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          width_ratios=[1.2, 1.0], hspace=0.55, wspace=0.28)
    a = fig.add_subplot(gs[0, 0])
    c = fig.add_subplot(gs[0, 1])
    b = fig.add_subplot(gs[1, :])

    # ---- A: the sign reversal. Effect against the teacher's disagreement rate.
    a.axhline(0, color="0.3", lw=0.8)
    for score, label, marker, colour in RULES:
        xs, ys = [], []
        for _, spec_name, disagree in CORPORA:
            if score in data[spec_name]:
                xs.append(disagree)
                ys.append(st.mean(data[spec_name][score]["folds"]))
        a.plot(xs, ys, marker=marker, color=colour, ms=4.5, lw=1.0, label=label)
    a.set_xlabel("teacher's top-1 differs from the reference")
    a.set_ylabel("accuracy vs. random control (points)")
    a.set_xticks([d for _, _, d in CORPORA])
    a.set_xticklabels([f"{n}\n{d}%" for n, _, d in CORPORA], fontsize=7.5)
    # The ordering holds on the three corpora it was drawn from and fails on
    # the first one it was asked to predict. Leaving that off the panel would
    # let the three lines assert a law the paper reports as broken, so the
    # out-of-sample point is drawn with the lines rather than only in the text.
    # A small red marker, named in the caption and discussed in the text; the
    # panel carries no second label for it.
    a.plot([ORCA_OOS[0]], [ORCA_OOS[1]], marker="o", ms=5, color="#c00000",
           ls="none", zorder=5)
    a.set_xlim(17, 72)
    # Headroom above the data so the legend has somewhere to sit that no line
    # passes through: every series peaks below +2.
    a.set_ylim(-5.2, 5.0)
    a.set_title("A   the ordering, and where it breaks", loc="left",
                fontsize=8, fontweight="bold")
    a.set_yticks([-4, -2, 0, 2])
    a.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2,
             handlelength=1.4, columnspacing=1.0, labelspacing=0.3,
             borderaxespad=0.2)

    # ---- B: every paired run, so the win count is printed for checking.
    b.axvline(0, color="0.3", lw=0.8)
    rows, labels = [], []
    for name, spec_name, _ in CORPORA:
        for score, label, _marker, colour in RULES:
            if score not in data[spec_name]:
                continue
            rows.append((data[spec_name][score], colour))
            labels.append(f"{name}  {label}")
    ys = range(len(rows))
    for y, (cell, colour) in zip(ys, rows):
        b.plot(cell["runs"], [y] * len(cell["runs"]), "o", ms=2.6, color=colour,
               alpha=0.55, mec="none")
        b.plot([st.mean(cell["folds"])], [y], "|", ms=11, mew=1.6, color=colour)
        b.annotate(f"{cell['wins']}/{len(cell['runs'])}", (7.4, y), fontsize=7,
                   va="center", color="0.25")
    b.set_yticks(list(ys))
    b.set_yticklabels(labels, fontsize=7)
    b.grid(axis="y", color="0.9", lw=0.5)
    b.set_axisbelow(True)
    b.invert_yaxis()
    b.set_xlim(-11, 9.4)
    b.set_xlabel("one run minus its paired control (points)")
    b.set_title("C   every paired run", loc="left", fontsize=8, fontweight="bold")

    # ---- C: the criterion survives truncation, so an API's top-K suffices.
    ks = [r[0] for r in TRUNCATION]
    c.plot(ks, [r[3] for r in TRUNCATION], "o-", color="#1f4e79", ms=4,
           label="same tenth selected")
    c.plot(ks, [100 * r[2] for r in TRUNCATION], "s--", color="#b45f06", ms=4,
           label="rank correlation $\\times\\,100$")
    c.set_xscale("log")
    c.set_xticks(ks)
    c.set_xticklabels([str(k) for k in ks])
    # A log axis draws its own minor labels on top of ours; hiding the ticks
    # is not enough, the formatter has to go too.
    c.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.tick_params(axis="x", which="minor", bottom=False)
    c.set_ylim(92, 100.6)
    c.set_xlabel("stored entries $K$ per position")
    c.set_ylabel("agreement with the full distribution (%)")
    c.set_title("B   top-$K$ suffices", loc="left", fontsize=8, fontweight="bold")
    c.legend(fontsize=7, frameon=False, loc="lower right", handlelength=1.8)

    fig.savefig(OUT / "fig1.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_budget():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    pct = [r[0] for r in BUDGET]
    ax.axhline(WHOLE_CORPUS, color="0.6", lw=0.8, ls=(0, (4, 3)))
    ax.annotate("whole corpus, text only", (20, WHOLE_CORPUS + 0.2),
                color="0.35", fontsize=6.5, ha="right")
    ax.axhline(BASE, color="0.6", lw=0.8, ls=(0, (1, 2)))
    ax.annotate("untrained base", (20, BASE + 0.2), color="0.35",
                fontsize=6.5, ha="right")
    ax.plot(pct, [r[3] for r in BUDGET], "o-", color="#1f4e79", ms=4,
            label="artifact")
    ax.plot(pct, [r[2] for r in BUDGET], "s-", color="#b45f06", ms=4,
            label="text only")
    ax.set_xscale("log")
    ax.set_xticks(pct)
    ax.set_xticklabels([f"{p}%" for p in pct])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("sequences used, as a fraction of the corpus")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.legend(fontsize=7, frameon=False, loc="center left")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_budget.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "dejavuserif"})
    data = {name: paired_runs(name) for _, name, _ in CORPORA}
    for name, spec_name, _ in CORPORA:
        for score, cell in data[spec_name].items():
            print(f"{name:6s} {score:14s} fold-mean {st.mean(cell['folds']):+6.2f}  "
                  f"wins {cell['wins']}/{len(cell['runs'])}")
    fig1(data)
    print(f"wrote {OUT/'fig1.pdf'}")
    # The budget curve is not a figure in the paper, which carries that
    # result in the text. Available behind a flag.
    if "--budget" in sys.argv:
        fig_budget()
        print(f"wrote {OUT/'fig_budget.pdf'}")


if __name__ == "__main__":
    main()
