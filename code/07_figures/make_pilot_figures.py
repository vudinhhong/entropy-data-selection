#!/usr/bin/env python3
"""Figs 2 to 4 of the paper: the two-ended contrast, and Pilot-Based
Direction Selection (PDS).

Both are drawn from `data/per_run_accuracies.csv` through the same two modules
that produce the corresponding tables, so a figure cannot disagree with the
table beside it:

  Fig 2   every contrast in Table "High-scoring minus low-scoring", with its
          detection floor, from code/16_contrast/two_ended_contrast.py
  Fig 3   PDS, from code/18_direction/direction_pilot.py

    python make_pilot_figures.py

Needs matplotlib. Writes beside the manuscript when there is one and into
figures/ otherwise, like make_figures.py.
"""
import importlib.util
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "paper" if (ROOT / "paper").is_dir() else ROOT / "figures"

# Fig 2 uses these two for the two ends of the scale, high and low.
HIGH, LOW, GREY = "#1f4e79", "#b45f06", "0.55"
# Fig 3 is about policies, not ends, so it uses a palette of its own: no
# reader should carry "orange means the low end" from Fig 2 into Fig 3.
DEFAULT, PILOT, ORACLE, RULE2 = "#8c8c8c", "#1f4e79", "#c9c9c9", "#6a3d9a"
TCRIT4 = 2.7764451051977987


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [name]            # the modules parse argv at import time
    spec.loader.exec_module(mod)
    return mod


def short(label):
    """Shorten a cell label for an axis tick."""
    return (label.replace(" / ", "\n").replace("Qwen1.5-1.8B", "Qwen1.5-1.8B")
            .replace(", untrained, full pool", ", untrained, full")
            .replace("-Math", ""))


# ----------------------------------------------------------------- figure 2

def fig_contrast(tec):
    by = tec.load()
    rows = []
    for label, scores, *groups in tec.CELLS:
        cells = {}
        for g in groups:
            cells.update(by[g])
        for which in scores:
            hi, lo = tec.ENDS[which]
            r = tec.contrast(cells, hi, lo)
            if r:
                rows.append((f"{label}\n{which} entropy", r))
    alpha = 0.05 / len(rows)
    rows.sort(key=lambda t: t[1]["mean"])

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for i, (label, r) in enumerate(rows):
        colour = HIGH if r["mean"] > 0 else LOW
        survives = r["p"] < alpha
        ax.errorbar(r["mean"], i, xerr=r["floor"], fmt="o", ms=5.5,
                    color=colour, ecolor=colour, elinewidth=1.4, capsize=2.5,
                    mfc=colour if survives else "white", mew=1.4, zorder=3)
    ax.axvline(0, color="0.3", lw=0.9, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([l for l, _ in rows], fontsize=6.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("high-scoring tenth minus low-scoring tenth (points)")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    # Anchored to the zero line in x, so each label sits on the side it names.
    from matplotlib.transforms import blended_transform_factory
    span = blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate("low end better", (-0.25, 1.015), xycoords=span, ha="right",
                fontsize=7, color=LOW)
    ax.annotate("high end better", (0.25, 1.015), xycoords=span, ha="left",
                fontsize=7, color=HIGH)
    ax.annotate(f"filled: survives Bonferroni at {alpha:.4f}\n"
                f"bars: detection floor, $t_{{0.975,4}}\\,s/\\sqrt{{5}}$",
                (0.985, 0.03), xycoords="axes fraction", fontsize=6.5,
                color="0.3", ha="right")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig2.pdf", bbox_inches="tight")
    plt.close(fig)
    return len(rows), sum(1 for _, r in rows if r["p"] < alpha)


# ----------------------------------------------------------------- figure 3

def panel_regret(ax, sigma):
    """R(Delta) and the constant that caps it."""
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
    xs = [i / 200 for i in range(0, 1201)]
    for m, style in ((2, "-"), (3, "--"), (4, ":")):
        ys = [x * phi(-math.sqrt(m) * x / sigma) for x in xs]
        ax.plot(xs, ys, style, color=HIGH, lw=1.4, label=f"$m = {m}$")
        cap = 0.17 * sigma / math.sqrt(m)
        ax.axhline(cap, color=GREY, lw=0.7, ls=(0, (1, 2)))
    ax.annotate("$0.17\\,\\sigma/\\sqrt{m}$", (3.4, 0.17 * sigma / math.sqrt(2)),
                fontsize=6.5, color="0.3", va="bottom")
    ax.set_xlabel("true contrast $|\\Delta|$ (points)")
    ax.set_ylabel("expected forfeit (points)")
    ax.set_xlim(0, 6)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def panel_policies(ax, dp, cells):
    rows = [("always high", dp.fixed(cells, "H")),
            ("always low", dp.fixed(cells, "L")),
            ("sign rule, 4 fine-tunes", dp.evaluate(cells, 2, 1, "sign")[0]),
            ("sign rule, 12 fine-tunes", dp.evaluate(cells, 3, 2, "sign")[0]),
            ("interval rule, 16 fine-tunes", dp.evaluate(cells, 4, 2, "interval")[0]),
            ("oracle", dp.fixed(cells, "oracle"))]
    names = [n for n, _ in rows]
    means = [dp.mean([v for _, v in r]) for _, r in rows]
    worst = [min(v for _, v in r) for _, r in rows]
    y = range(len(rows))
    colours = [DEFAULT, DEFAULT, PILOT, PILOT, RULE2, ORACLE]
    ax.barh(list(y), means, color=colours, height=0.6, zorder=2)
    ax.plot(worst, list(y), "|", color="0.25", ms=9, mew=1.4, zorder=3,
            label="worst cell")
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean gain over the random control (points)")
    # Outside the axes: inside, a lone marker key sits at a plausible value
    # on some row and reads as a data point.
    ax.legend(fontsize=6.5, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), borderaxespad=0.1)
    ax.spines[["top", "right"]].set_visible(False)


def panel_corpora(ax, dp, cells):
    pilot = dict(dp.evaluate(cells, 2, 1, "sign")[0])
    high, orac = dict(dp.fixed(cells, "H")), dict(dp.fixed(cells, "oracle"))
    groups = {}
    for key in pilot:
        groups.setdefault(key.split("/")[0].strip(), []).append(key)
    order = sorted(groups, key=lambda c: -len(groups[c]))
    width, xs = 0.27, range(len(order))
    for off, (src, colour, name) in enumerate(
            ((high, DEFAULT, "always the high end"),
             (pilot, PILOT, "pilot, 4 fine-tunes"),
             (orac, ORACLE, "oracle direction"))):
        ax.bar([x + (off - 1) * width for x in xs],
               [dp.mean([src[k] for k in groups[c]]) for c in order],
               width, color=colour, label=name, zorder=2)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{c}\n({len(groups[c])})" for c in order], fontsize=6.5)
    ax.set_ylabel("gain over random (points)")
    # Back to the lower right, where it was verified clear. Upper left looks
    # free in the standalone panel but not in the composite, where the axis
    # is shorter and the ECQA bar reaches into it.
    # Hard against the right spine and narrowed, so its left edge clears the
    # ECQA bar. Checked in the composite, not in the standalone panel.
    ax.legend(fontsize=6, frameon=False, loc="lower right", borderaxespad=0.0,
              handlelength=1.1, handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def panel_cost(ax, dp, cells):
    for rule, colour, marker, name in (
            ("sign", PILOT, "o", "sign of the contrast"),
            ("interval", RULE2, "s", "only if significant")):
        pts = []
        for m in (2, 3, 4):
            for seeds in (1, 2):
                per, _, _ = dp.evaluate(cells, m, seeds, rule)
                pts.append((2 * m * seeds, dp.mean([v for _, v in per])))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker + "-",
                color=colour, ms=4.5, lw=1.3, label=name)
    ax.axhline(dp.mean([v for _, v in dp.fixed(cells, "oracle")]),
               color="0.5", lw=0.8, ls=(0, (4, 3)))
    ax.annotate("oracle direction",
                (16, dp.mean([v for _, v in dp.fixed(cells, "oracle")])),
                fontsize=6.5, color="0.35", va="bottom", ha="right")
    ax.annotate("two ways\nto spend 8", (7.7, 0.30), fontsize=6, color="0.4",
                ha="right", va="top")
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_xticks([4, 6, 8, 12, 16])
    ax.set_xlabel("fine-tunes the pilot spends")
    ax.set_ylabel("mean gain (points)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 0.04))
    ax.spines[["top", "right"]].set_visible(False)


def panel_bound(ax, dp, cells):
    rows = dp.regret(cells, 2, 1)
    lim = max(max(r[1] for r in rows), max(r[2] for r in rows)) * 1.12
    ax.plot([0, lim], [0, lim], color="0.6", lw=0.9, ls=(0, (4, 3)), zorder=1)
    over = [r for r in rows if r[1] > r[2]]
    ax.scatter([r[2] for r in rows if r not in over],
               [r[1] for r in rows if r not in over],
               s=22, color=PILOT, zorder=3, label="within its bound")
    ax.scatter([r[2] for r in over], [r[1] for r in over], s=22, color=LOW,
               zorder=3, label=f"above it ({len(over)}/{len(rows)})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("$0.17\\,\\hat{\\sigma}/\\sqrt{m}$ for that cell (points)")
    ax.set_ylabel("regret against the oracle (points)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 0.12))
    ax.spines[["top", "right"]].set_visible(False)


def fig_pilot(dp):
    cells = dp.load("acc_lenient")
    sigma = st.median(dp.spread(cells, 1))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    panel_regret(axes[0][0], sigma)
    panel_policies(axes[0][1], dp, cells)
    panel_corpora(axes[1][0], dp, cells)
    panel_cost(axes[1][1], dp, cells)
    for ax, letter in zip(axes.ravel(), "ABCD"):
        ax.annotate(letter, (-0.17, 1.04), xycoords="axes fraction",
                    fontweight="bold", fontsize=10)
    fig.tight_layout(pad=0.6, w_pad=2.0, h_pad=1.6)
    fig.savefig(OUT / "fig3.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    panel_bound(ax, dp, cells)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig4.pdf", bbox_inches="tight")
    plt.close(fig)
    return len(cells), sigma


def panels(dp):
    """The four panels of Fig 3 as separate files, for slides.

    A talk shows one panel at a time; cropping the composite is brittle, so
    each panel is drawn again on its own canvas from the same functions.
    """
    cells = dp.load("acc_lenient")
    sigma = st.median(dp.spread(cells, 1))
    jobs = [("fig3a", lambda ax: panel_regret(ax, sigma), (4.0, 3.0)),
            ("fig3b", lambda ax: panel_policies(ax, dp, cells), (4.6, 3.0)),
            ("fig3c", lambda ax: panel_corpora(ax, dp, cells), (4.6, 3.0)),
            ("fig3d", lambda ax: panel_cost(ax, dp, cells), (4.2, 3.0))]
    for name, draw, size in jobs:
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        fig.tight_layout(pad=0.4)
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {OUT/'fig3a.pdf'} .. {OUT/'fig3d.pdf'} for slides")


def main():
    # module() replaces sys.argv while importing, so read the flag first.
    want_panels = "--panels" in sys.argv
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 300,
                         "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    tec = module("two_ended_contrast", ROOT / "code" / "16_contrast"
                 / "two_ended_contrast.py")
    n, survive = fig_contrast(tec)
    print(f"wrote {OUT/'fig2.pdf'}: {n} contrasts, {survive} survive Bonferroni")

    dp = module("direction_pilot", ROOT / "code" / "18_direction"
                / "direction_pilot.py")
    cells, sigma = fig_pilot(dp)
    print(f"wrote {OUT/'fig3.pdf'} and {OUT/'fig4.pdf'}: {cells} cells, "
          f"median one-seed spread {sigma:.2f}")
    if want_panels:
        panels(dp)


if __name__ == "__main__":
    main()
