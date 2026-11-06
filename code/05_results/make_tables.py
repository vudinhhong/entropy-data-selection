"""Every table in the paper, rebuilt from the raw result files.

No GPU, no models, no corpus download. Each file in `results/grid/` carries the
accuracy, the training settings and the generated text for all 1,319 (GSM8K) or
500 (MATH) test items, which is enough to recompute an accuracy, a paired test,
a per-subset spread or a McNemar reading from scratch.

    python make_tables.py
    python make_tables.py --table cells --table sham

`config.json` names, for each cell and each comparison, the exact result files
that produce it. Nothing is matched by pattern, so a table cannot quietly pick
up a run that was never meant to be in it.

A word on the statistics, because the two readings here disagree and the paper
takes a side. The unit of generalisation for a claim about a *method* is the
training subset: three subsets, paired by subset, df = 2, critical t = 4.303.
McNemar pooled over subsets reuses the same test items three times and is
correspondingly anti-conservative. Both are printed. Where they disagree, the
subset test is the one the paper reports.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdstore import Report, load_config
from kdstore.paths import resolve
from kdstore.stats import (discordant, mcnemar, paired_t_test, pearson,
                           qualifies, se_bound)

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- loading

class Results:
    """The raw result files, read once and indexed by tag."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self._cache = {}

    def get(self, tag):
        if tag not in self._cache:
            path = self.directory / f"grid_{tag}.json"
            if not path.exists():
                path = self.directory / f"{tag}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"no result file for tag {tag!r} in {self.directory}")
            self._cache[tag] = json.loads(path.read_text())
        return self._cache[tag]

    def accuracy(self, tag, metric="lenient"):
        key = "acc_strict" if metric == "strict" else "acc"
        return self.get(tag)[key] * 100

    def flags(self, tag, metric="lenient"):
        """Per-item correctness, for McNemar."""
        key = "strict" if metric == "strict" else "ok"
        return [bool(g[key]) for g in self.get(tag).get("gens", [])]

    def arm(self, tags, metric="lenient"):
        return [self.accuracy(t, metric) for t in tags]

    def arm_or_none(self, tags, metric="lenient"):
        """An arm, or None if any of its points has not been run yet.

        Tables are often read while the experiment is still going, and a
        half-finished arm is worse than an absent one: averaging the points
        that happen to have landed first reports a number nobody asked for.
        """
        try:
            return self.arm(tags, metric)
        except FileNotFoundError:
            return None


def summarise(values):
    import statistics
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}


# ---------------------------------------------------------------- tables

def table_qualification(config, results, report):
    """Which cells the validity criterion admits, and why.

    A cell is readable only if fine-tuning on the whole student half moves the
    base model by more than twice the evaluation standard error. Below that
    the measurement has no room to show a difference between arms, so a null
    result there says nothing about the method, it says the instrument was
    not sensitive enough to be worth reading.

    The standard error is the worst-case bound sqrt(0.25/n), not the realised
    value. A threshold that moves with the measurement is one that can be
    argued with afterwards.
    """
    rows = []
    for name, cell in config["cells"].items():
        dataset = cell["dataset"]
        n_eval = config["eval_size"][dataset]
        verdict = qualifies(cell["base"], cell["full_corpus"], n_eval)
        rows.append([name, cell["base"], cell["full_corpus"], verdict["gain"],
                     verdict["threshold"],
                     "qualified" if verdict["qualified"] else "excluded"])
    report.table("qualification",
                 ["cell", "base", "whole corpus", "gain", "2 SE", "verdict"],
                 rows, title="Two-sided validity criterion")


def table_cells(config, results, report):
    """The three-arm comparison on every qualified cell."""
    rows = []
    recorded = {}
    for name, cell in config["cells"].items():
        dataset = cell["dataset"]
        n_eval = config["eval_size"][dataset]
        if not qualifies(cell["base"], cell["full_corpus"],
                         n_eval)["qualified"]:
            continue

        text = results.arm(cell["arms"]["text"])
        artifact = results.arm(cell["arms"]["artifact"])
        low = results.arm(cell["arms"].get("lowgate", []))
        paired = paired_t_test([a - t for a, t in zip(artifact, text)])

        # Both arms are fine-tuned, so both have learned the answer format and
        # the strict metric is a fair comparison between them. (It is not a
        # fair input to the qualification criterion: an untrained base scores
        # near zero strict whatever it knows, because it never emits the
        # marker.)
        text_s = results.arm(cell["arms"]["text"], "strict")
        artifact_s = results.arm(cell["arms"]["artifact"], "strict")
        paired_s = paired_t_test([a - t for a, t in zip(artifact_s, text_s)])

        teacher = config["teacher"][dataset]
        denominator = teacher - cell["base"]
        pvr_artifact = (summarise(artifact)["mean"] - cell["base"]) / denominator
        pvr_corpus = (cell["full_corpus"] - cell["base"]) / denominator

        rows.append([name, summarise(text)["mean"],
                     summarise(artifact)["mean"],
                     summarise(low)["mean"] if low else None,
                     paired["mean"], paired["sd"], paired["t"], paired["p"],
                     paired_s["mean"], paired_s["sd"], paired_s["p"],
                     f"{pvr_artifact * 100:.1f}%", f"{pvr_corpus * 100:.1f}%"])
        recorded[name] = {"text": text, "artifact": artifact, "lowgate": low,
                          "paired": paired, "paired_strict": paired_s,
                          "pvr": pvr_artifact}

    report.table("cells",
                 ["cell", "text", "artifact", "low gate", "effect", "sd",
                  "t", "p", "effect (strict)", "sd", "p (strict)",
                  "PVR artifact", "PVR corpus"],
                 rows, title="Qualified cells, three subsets per arm "
                             "(lenient, with the strict reading alongside)",
                 places=3)
    report.data["values"]["cells"] = recorded
    return recorded


def table_disqualified(config, results, report):
    """Cells the criterion excludes, run anyway.

    This is the check on the criterion itself. If it were a device for
    discarding inconvenient results it would be excluding cells where the
    artifact loses. It is not: both excluded cells show the artifact ahead of
    text, one of them at p = 0.009, smaller than five of the six qualified
    cells. What makes them unreadable is that every arm, artifact included,
    finishes *below* the untrained base model, so the ordering between arms
    cannot be read as evidence about the method. The criterion excludes
    misreadable evidence, not unwelcome evidence.
    """
    rows = []
    for name, cell in config["cells"].items():
        dataset = cell["dataset"]
        if qualifies(cell["base"], cell["full_corpus"],
                     config["eval_size"][dataset])["qualified"]:
            continue
        arms = cell.get("arms", {})
        if "text" not in arms or "artifact" not in arms:
            rows.append([name, cell["base"], None, None, None, None, None,
                         "no arms run"])
            continue
        text = results.arm(arms["text"])
        artifact = results.arm(arms["artifact"])
        paired = paired_t_test([a - t for a, t in zip(artifact, text)])
        below = max(summarise(text)["mean"],
                    summarise(artifact)["mean"]) < cell["base"]
        rows.append([name, cell["base"], summarise(text)["mean"],
                     summarise(artifact)["mean"], paired["mean"],
                     paired["sd"], paired["p"],
                     "every arm below base" if below else "mixed"])

    if rows:
        report.table("disqualified",
                     ["cell", "base", "text", "artifact", "effect", "sd",
                      "p", "why it is unreadable"],
                     rows, title="Excluded cells, run deliberately",
                     places=3)


def table_two_readings(config, results, report):
    """The subset test and the pooled McNemar, for every arm comparison.

    They disagree, and the disagreement is the point. McNemar asks whether the
    artifact arm beats the text arm on these particular test items; the paired
    subset test asks whether it would beat it on a fresh draw of training data.
    Only the second is a claim about the method, and the two come apart in a
    predictable direction: pooling discordant pairs over subsets counts the
    same test items three times, so McNemar is the one that over-claims.

    Every comparison is listed, including the ones on cells the criterion
    excludes, because a table that showed only the headline comparison would
    not let a reader check that assertion.
    """
    rows = []
    disagreements = 0
    for name, cell in config["cells"].items():
        arms = cell.get("arms", {})
        pairs = []
        if "text" in arms and "artifact" in arms:
            pairs.append(("artifact - text", arms["artifact"], arms["text"]))
        if "lowgate" in arms:
            pairs.append(("artifact - low gate", arms["artifact"],
                          arms["lowgate"]))
            pairs.append(("low gate - text", arms["lowgate"], arms["text"]))

        qualified = qualifies(cell["base"], cell["full_corpus"],
                              config["eval_size"][cell["dataset"]])["qualified"]
        for label, a_tags, b_tags in pairs:
            differences, b_total, c_total = [], 0, 0
            for a_tag, b_tag in zip(a_tags, b_tags):
                differences.append(results.accuracy(a_tag)
                                   - results.accuracy(b_tag))
                a_flags, b_flags = results.flags(a_tag), results.flags(b_tag)
                if a_flags and b_flags:
                    x, y = discordant(a_flags, b_flags)
                    b_total += x
                    c_total += y

            paired = paired_t_test(differences)
            pooled = mcnemar(b_total, c_total)
            agree = (paired["p"] < 0.05) == (pooled["p"] < 0.05)
            disagreements += not agree
            rows.append([name + ("" if qualified else " (excluded)"), label,
                         paired["mean"], paired["p"], b_total, c_total,
                         pooled["p"], "agree" if agree else "DISAGREE"])

    report.table("two_readings",
                 ["cell", "comparison", "effect", "subset p", "first only",
                  "second only", "McNemar p", "readings"],
                 rows, title="Subset-level test against pooled McNemar, "
                             "every comparison", places=4)
    report.value("disagreements", disagreements,
                 f"disagreements out of {len(rows)} comparisons")
    # Every disagreement should run the same way if pooling is what inflates
    # McNemar; say so explicitly instead of leaving it to be eyeballed.
    one_way = all(r[6] < 0.05 <= r[3] for r in rows if r[7] == "DISAGREE")
    report.value("disagreements_all_mcnemar_significant", one_way,
                 "every disagreement is McNemar-significant, subset-test not")


def table_cross_cell(name, spec, config, results, report):
    """A claim tested on several cells at once.

    Two readings are printed and they are not interchangeable. Pooling every
    subset treats the subset as the unit, which is right within a cell but
    counts subsets from the same cell as if they were independent of each
    other. Taking the cell as the unit asks the harder question, would this
    hold on a cell we have not run, and with a handful of cells it has very
    few degrees of freedom. Where the two disagree, the cell-level reading is
    the one a claim about generality has to survive.
    """
    rows, cell_means, pooled = [], [], []
    for cell, arms in spec["cells"].items():
        a = results.arm_or_none(arms["a"])
        b = results.arm_or_none(arms["b"])
        if a is None or b is None:
            rows.append([cell, None, None, None, None, None, None])
            continue
        differences = [x - y for x, y in zip(a, b)]
        cell_means.append(sum(differences) / len(differences))
        pooled += differences
        paired = paired_t_test(differences)
        rows.append([cell, len(differences), summarise(b)["mean"],
                     summarise(a)["mean"], paired["mean"], paired["sd"],
                     paired["p"]])

    report.table(name, ["cell", "n", "without", "with", "difference", "sd",
                        "p"], rows,
                 title=spec.get("description", name), places=3)

    by_cell = paired_t_test(cell_means)
    by_subset = paired_t_test(pooled)
    report.table(f"{name}_readings",
                 ["unit of generalisation", "n", "effect", "sd", "t", "p"],
                 [["cell", by_cell["n"], by_cell["mean"], by_cell["sd"],
                   by_cell["t"], by_cell["p"]],
                  ["subset (pooled across cells)", by_subset["n"],
                   by_subset["mean"], by_subset["sd"], by_subset["t"],
                   by_subset["p"]]],
                 title="Two readings of the same numbers", places=4)


def table_folds(name, spec, config, results, report):
    """A selection criterion tested with the fold as the unit.

    The scored arms are deterministic on a fixed pool, so running one selected
    subset under several training seeds produces several numbers but only one
    observation of the criterion: the seeds vary the optimiser, not the
    choice. A paired test over those seeds would treat one subset as many and
    report a confidence the design does not support.

    This table pairs by fold instead. Each fold is a disjoint slice of the
    pool, selected independently by every score, so one fold contributes one
    difference and the degrees of freedom are the number of folds minus one.
    Training seeds within a fold are averaged first, which is what they are
    for, they reduce the noise inside an observation instead of adding
    observations.
    """
    scores = spec["scores"]
    control = spec.get("control", "random")
    folds, missing = {}, []
    for f in range(spec["folds"]):
        per_score = {}
        for score in scores:
            tags = [spec["tag"].format(fold=f, score=score, seed=seed)
                    for seed in spec["seeds"]]
            values = results.arm_or_none(tags)
            if values is None:
                missing.append(f"fold {f}, {score}")
                continue
            per_score[score] = sum(values) / len(values)
        if len(per_score) == len(scores):
            folds[f] = per_score

    rows = []
    for f, per_score in sorted(folds.items()):
        rows.append([f] + [round(per_score[s], 2) for s in scores])
    report.table(f"{name}_folds", ["fold", *scores], rows,
                 title=f"{spec['description']}: per-fold means")

    contrasts = []
    for score in scores:
        if score == control:
            continue
        differences = [folds[f][score] - folds[f][control] for f in sorted(folds)]
        if len(differences) < 2:
            contrasts.append([f"{score} - {control}", len(differences),
                              None, None, None])
            continue
        paired = paired_t_test(differences)
        contrasts.append([f"{score} - {control}", len(differences),
                          round(paired["mean"], 2), round(paired["sd"], 2),
                          round(paired["p"], 4)])
    for pair in spec.get("extra_contrasts", []):
        a, b = pair
        differences = [folds[f][a] - folds[f][b] for f in sorted(folds)]
        if len(differences) >= 2:
            paired = paired_t_test(differences)
            contrasts.append([f"{a} - {b}", len(differences),
                              round(paired["mean"], 2), round(paired["sd"], 2),
                              round(paired["p"], 4)])
    report.table(f"{name}_contrasts",
                 ["contrast", "folds", "mean", "s.d.", "p"], contrasts,
                 title=f"{spec['description']}: paired by fold")
    if missing:
        report.note(f"  {name}: not yet run -- " + "; ".join(missing))


def table_comparison(name, spec, config, results, report):
    """A generic one-factor comparison: several arms, same subsets.

    Used for the sham stores, the width sweep, the temperature check and the
    disqualified cells. The baseline arm, if the spec names one, is subtracted
    per subset so the test stays paired.
    """
    baseline = spec.get("baseline")
    base_values = (results.arm_or_none(spec["arms"][baseline])
                   if baseline else None)

    rows = []
    for arm, tags in spec["arms"].items():
        values = results.arm_or_none(tags)
        if values is None:
            rows.append([arm, None, None, None, None, None, None, None])
            continue
        stats = summarise(values)
        row = [arm, stats["n"], stats["mean"], stats["sd"]]
        if base_values and arm != baseline:
            paired = paired_t_test(
                [v - b for v, b in zip(values, base_values)])
            row += [paired["mean"], paired["sd"], paired["t"], paired["p"]]
        else:
            row += [None, None, None, None]
        rows.append(row)

    report.table(name,
                 ["arm", "n", "mean", "sd", f"vs {baseline}", "sd", "t", "p"],
                 rows, title=spec.get("description", name), places=3)


def table_correlations(config, results, report, cells):
    """Does a cheap property of the base model predict the gain?

    Both series must come from the same corpus. Pairing a calibration number
    measured on one corpus with a gain measured on another produces a
    correlation that disappears once the corpus is matched.
    """
    measured = config.get("base_model_properties")
    if not measured:
        return
    rows = []
    for quantity in next(iter(measured.values())).keys():
        xs, ys = [], []
        for name, stats in cells.items():
            if name in measured:
                xs.append(measured[name][quantity])
                ys.append(stats["paired"]["mean"])
        if len(xs) >= 3:
            r = pearson(xs, ys)
            rows.append([quantity, r["n"], r["r"], r["p"],
                         f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"])
    if rows:
        report.table("correlations",
                     ["property of the base model", "n", "Pearson r", "p",
                      "95% CI"], rows,
                     title="Corpus-matched, over the qualified cells",
                     places=4)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--table", action="append",
                    help="repeatable; default is every table")
    args = ap.parse_args()

    config = load_config(args.config)
    results = Results(resolve(config.get("results_dir", "results/grid")))
    wanted = args.table

    report = Report("Results")

    def want(name):
        return not wanted or name in wanted

    cells = {}
    if want("qualification"):
        table_qualification(config, results, report)
    if want("cells"):
        cells = table_cells(config, results, report)
    if want("disqualified"):
        table_disqualified(config, results, report)
    if want("two_readings"):
        table_two_readings(config, results, report)
    for name, spec in config.get("comparisons", {}).items():
        if want(name):
            table_comparison(name, spec, config, results, report)
    for name, spec in config.get("folds", {}).items():
        if want(name):
            table_folds(name, spec, config, results, report)
    for name, spec in config.get("cross_cell", {}).items():
        if want(name):
            table_cross_cell(name, spec, config, results, report)
    if want("correlations") and cells:
        table_correlations(config, results, report, cells)

    report.note()
    report.note(f"  evaluation standard-error bound: "
                f"GSM8K {se_bound(config['eval_size']['gsm8k']):.2f}, "
                f"MATH {se_bound(config['eval_size']['math']):.2f} points")

    out = resolve(config.get("report_dir", "results"))
    stem = "tables"
    if args.table:
        stem += "_" + "-".join(sorted(args.table))
    report.save(out / f"{stem}.json", out / f"{stem}.txt")


if __name__ == "__main__":
    main()
