#!/usr/bin/env python3
"""Does run_pilot.py still agree with the stages it drives?

run_pilot.py writes the configuration that build_store.py and
run_experiments.py read, then reads back the files they write. Three names
have to match across the three scripts, and nothing in Python checks them:

  * the subset file the fold selection writes, which the training config names;
  * the tag the training expands to, which becomes the result filename;
  * the accuracy keys inside that result file.

A rename on either side leaves run_pilot.py writing a configuration that runs
and produces nothing it can find. This test reads the naming rules out of the
two stages and compares them with what run_pilot.py expects, then exercises
the decision on results it fabricates. Standard library only, no GPU.

    python tests/test_pilot_config.py
"""
import argparse
import importlib.util
import json
import math
import re
import sys
import tempfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(label, got, want):
    if got != want:
        print(f"  FAIL {label}\n       run_pilot: {got}\n       stage:     {want}")
        return 1
    print(f"  ok   {label}: {got}")
    return 0


def subset_rule():
    """The filename build_store.py gives a fold selection."""
    source = (CODE / "01_build_store" / "build_store.py").read_text()
    m = re.search(r'name = f"\{cfg\[.prefix.\]\}(.+?)"', source)
    if not m:
        raise SystemExit("could not find the subset naming rule in build_store.py")
    return "{prefix}" + m.group(1).replace("{f}", "{fold}").replace(
        "{score}", "{score}")


def tag_rule():
    """The tag run_experiments.py builds a result filename from."""
    source = (CODE / "02_train_students" / "run_experiments.py").read_text()
    m = re.search(r'point\["tag"\] = "_".join\(\s*str\(p\) for p in \((.+?)\)\)',
                  source, re.S)
    if not m:
        raise SystemExit("could not find the tag rule in run_experiments.py")
    # Split on top-level commas only: the first field is group.get("prefix",
    # name), which carries a comma of its own.
    fields, depth, current = [], 0, ""
    for ch in m.group(1):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            fields.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        fields.append(current.strip())
    return fields


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    pilot = load("run_pilot", CODE / "18_direction" / "run_pilot.py")
    bad = 0

    print("naming rules")
    rule = subset_rule()
    expected = "{prefix}{fold}_{score}"
    bad += check("subset file", expected, rule)

    fields = tag_rule()
    bad += check("tag fields", 4, len(fields))
    if not fields[0].startswith("group.get"):
        print(f"  FAIL tag starts with {fields[0]}, expected the group prefix")
        bad += 1

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        args = argparse.Namespace(
            folds=2, budget=75, coverage=0.297, seed=1234, device="cpu",
            teacher="T", student="S", min_share=0.015, k_max=50,
            store_batch_size=8, lora=32, lr=2e-4, epochs=2, batch_size=8,
            accumulate=2, eval_batch_size=24, max_new_tokens=320,
            metric="acc_strict")

        build = pilot.build_config(args, out, 400)
        fs = build["experiments"][0]["fold_selection"]
        print("\ngenerated configuration")
        bad += check("fold selection keeps both ends",
                     fs["scores"], ["gated_H", "gated_H_low"])
        bad += check("budget as a fraction of a fold", fs["fraction"], 75 / 200)
        bad += check("gate set by coverage", fs["tau"], "coverage:0.297")

        train = pilot.train_config(args, out)
        named = [Path(s["path"]).name for s in train["groups"]["pilot"]["subsets"]]
        want = [f"pilot{f}_{score}.json" for f in range(2)
                for score in ("gated_H", "gated_H_low")]
        bad += check("training names the files the selection writes", named, want)

        # The decision reads results/<tag>.json. Build those tags the way
        # run_experiments.py would, from the config just generated.
        group = train["groups"]["pilot"]
        tags = [f"{group['prefix']}_{st['label']}_{arm['name']}_{sub['label']}"
                for st in group["students"] for arm in group["arms"]
                for sub in group["subsets"]]
        results = out / "results"
        results.mkdir()
        # fold 0 favours the high end by 3 points, fold 1 by 1 point.
        accuracies = {"f0hi": 0.40, "f0lo": 0.37, "f1hi": 0.41, "f1lo": 0.40}
        for tag in tags:
            label = tag.rsplit("_", 1)[1]
            (results / f"{tag}.json").write_text(json.dumps(
                {"tag": tag, "acc": accuracies[label] + 0.01,
                 "acc_strict": accuracies[label]}))

        print("\ndecision on fabricated results")
        report = pilot.decide(args, out)
        bad += check("direction", report["direction"], "high")
        bad += check("score", report["score"], "gated_H")
        bad += check("mean contrast", round(report["mean_contrast"], 4), 2.0)
        sd = math.sqrt(((3.0 - 2.0) ** 2 + (1.0 - 2.0) ** 2) / 1)
        bad += check("regret bound",
                     round(report["expected_regret_bound"], 4),
                     round(0.17 * sd / math.sqrt(2), 4))
        bad += check("direction.json written",
                     (out / "direction.json").exists(), True)

    print(f"\n{'FAILED' if bad else 'all checks pass'}"
          + (f": {bad} problem(s)" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
