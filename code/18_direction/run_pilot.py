#!/usr/bin/env python3
"""Run Pilot-Based Direction Selection (PDS), Algorithm 1 of the paper,
on a corpus of your own.

The paper's claim is that which end of the entropy scale to select from is a
property of the corpus and has to be measured. This script measures it. Given
a training corpus, a held-out set to score on, a teacher and a student, it
runs one teacher forward pass, cuts the corpus into m folds, selects the top
and the bottom of each fold at the budget you will train at, fine-tunes one
student on each, and reports the direction together with the bound on what
choosing it wrongly can cost.

    python run_pilot.py --pairs train.jsonl --eval-pairs test.jsonl \\
        --teacher Qwen/Qwen3-4B-Base --student Qwen/Qwen1.5-1.8B \\
        --budget 75 --folds 2 --out pilot/

Input files are JSONL with a "question" and an "answer" field per line, or
JSON in the {"pairs": [[question, answer], ...]} form the rest of this code
already reads. The default of two folds at one seed is four fine-tunes, which
is the configuration Table 9 of the paper reports, scored by the same lenient
metric. Pass --metric acc_strict if the student may not learn the answer
format at this budget: the paper's Measurement floors section shows the
lenient metric crediting incidental matches when it has not.

The work is done by the two stages that produced every number in the paper:
this script writes the configuration they read, runs them, and interprets
the result. It never reimplements them. Run it with --plan to write the
configuration and print the commands without executing anything.

    python run_pilot.py ... --plan       write the configs, run nothing
    python run_pilot.py ... --decide     skip to the decision, using results
                                         already on disk

What it does not do: choose a metric for you. Students are scored by the
exact-match rule in kdstore/evaluation.py, written for answers that end in a
number or a \\boxed{} expression. A corpus whose answers look different needs
its own scorer, and the comparison is only as meaningful as that scorer.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent

# The maximum of x*Phi(-x), which bounds the expected cost of reading the
# sign of the contrast wrongly. See Eq 6 of the paper.
REGRET_CONSTANT = 0.17


# ------------------------------------------------------------------- inputs

def read_pairs(path):
    """Read (question, answer) pairs from JSONL or from a written-pairs JSON."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        pairs = []
        for n, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"question", "answer"} - set(row)
            if missing:
                raise SystemExit(f"{path}:{n}: line is missing {sorted(missing)}")
            pairs.append([row["question"], row["answer"]])
        return pairs
    payload = json.loads(text)
    if isinstance(payload, dict) and "pairs" in payload:
        return [list(p) for p in payload["pairs"]]
    raise SystemExit(f"{path}: expected JSONL, or JSON holding a 'pairs' list")


def write_pairs(pairs, path):
    path.write_text(json.dumps({"pairs": pairs}, ensure_ascii=False))
    return path


# ------------------------------------------------------- generated configs

def build_config(args, out, n_pairs):
    """The config the teacher pass and the fold selection read."""
    fold_size = n_pairs // args.folds
    if fold_size < args.budget:
        raise SystemExit(
            f"a fold holds {fold_size} sequences and the budget is "
            f"{args.budget}: either lower --budget, lower --folds, or bring "
            f"more data. Selection cannot choose from fewer than it keeps.")
    return {
        "report_dir": str(out / "reports"),
        "defaults": {"device": args.device, "seed": args.seed},
        "experiments": [{
            "name": "pilot",
            "artifacts": str(out / "subsets"),
            "store": {
                "teacher": args.teacher,
                "tokenizer": args.teacher,
                "corpus_file": str(out / "corpus.json"),
                "output": str(out / "store"),
                "k_max": args.k_max,
                "batch_size": args.store_batch_size,
            },
            "fold_selection": {
                "store": str(out / "store.npz"),
                "meta": str(out / "store.meta.json"),
                "pool": str(out / "pool.json"),
                "tau": f"coverage:{args.coverage}",
                "folds": args.folds,
                "fold_seed": args.seed,
                "fraction": args.budget / fold_size,
                "scores": ["gated_H", "gated_H_low"],
                "prefix": "pilot",
                "min_share": args.min_share,
            },
        }],
    }


def train_config(args, out):
    """The config the student training reads, one point per fold and end."""
    subsets = [{"path": str(out / "subsets" / f"pilot{f}_{score}.json"),
                "label": f"f{f}{'hi' if score == 'gated_H' else 'lo'}"}
               for f in range(args.folds)
               for score in ("gated_H", "gated_H_low")]
    return {
        "results_dir": str(out / "results"),
        "report_dir": str(out / "reports"),
        "defaults": {
            "device": args.device, "seed": args.seed, "k": 0,
            "corpus_file": str(out / "corpus.json"),
            "eval_file": str(out / "evalset.json"),
            "lora": args.lora, "lr": args.lr, "epochs": args.epochs,
            "batch_size": args.batch_size, "accumulate": args.accumulate,
            "eval_batch_size": args.eval_batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
        "groups": {"pilot": {
            "description": "PDS: both ends of each pilot fold",
            "prefix": "pilot",
            "students": [{"label": "s", "model": args.student}],
            "arms": [{"name": "text", "k": 0}],
            "subsets": subsets,
        }},
    }


# ------------------------------------------------------------------ running

def run(cmd, plan):
    printable = " ".join(str(c) for c in cmd)
    if plan:
        print(f"  {printable}")
        return
    print(f"\n$ {printable}", flush=True)
    if subprocess.run(cmd).returncode:
        raise SystemExit("stage failed; nothing was decided")


# ----------------------------------------------------------------- decision

def decide(args, out):
    """Read the trained points and report the direction."""
    results = out / "results"
    deltas = []
    for f in range(args.folds):
        acc = {}
        for end, label in (("H", "hi"), ("L", "lo")):
            path = results / f"pilot_s_text_f{f}{label}.json"
            if not path.exists():
                raise SystemExit(f"missing {path}; run without --decide first")
            acc[end] = 100 * json.loads(path.read_text())[args.metric]
        deltas.append(acc["H"] - acc["L"])

    m = len(deltas)
    bar = sum(deltas) / m
    direction = "high" if bar > 0 else "low"
    score = "gated_H" if bar > 0 else "gated_H_low"

    print(f"\nFold contrasts, high end minus low end, in points of {args.metric}")
    for f, d in enumerate(deltas):
        print(f"  fold {f}: {d:+.2f}")
    print(f"  mean   : {bar:+.2f}")

    report = {"direction": direction, "score": score, "metric": args.metric,
              "contrasts": deltas, "mean_contrast": bar, "folds": m}
    if m > 1:
        sd = math.sqrt(sum((d - bar) ** 2 for d in deltas) / (m - 1))
        bound = REGRET_CONSTANT * sd / math.sqrt(m)
        report.update(fold_sd=sd, expected_regret_bound=bound)
        print(f"  fold sd: {sd:.2f}")
        print(f"\nExpected cost of reading this sign wrongly is at most "
              f"{bound:.2f} points, 0.17 sd / sqrt(m).")
        print("Adding folds lowers it as 1/sqrt(m); the paper found the step "
              "changed nothing on its corpora.")
    else:
        print("\nOne fold gives no spread, so no bound can be stated. "
              "Use --folds 2 or more.")

    print(f"\nSelect from the {direction} end: {score}")
    path = out / "direction.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"Written to {path}")
    print("\nTo build the final selection over the whole corpus at your "
          "training budget, run the fold selection again with --folds 1 and "
          "the fraction you want, keeping only this score.")
    return report


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True,
                    help="training corpus, JSONL or written-pairs JSON")
    ap.add_argument("--eval-pairs", required=True,
                    help="held-out set the two arms are scored on")
    ap.add_argument("--out", required=True, help="directory for everything")
    ap.add_argument("--teacher", default="Qwen/Qwen3-4B-Base",
                    help="any model id or local path; it is never fine-tuned")
    ap.add_argument("--student", default="Qwen/Qwen1.5-1.8B")
    ap.add_argument("--budget", type=int, default=75,
                    help="sequences kept per fold; use the budget you will train at")
    ap.add_argument("--folds", type=int, default=2,
                    help="pilot folds; 2 is four fine-tunes and is the default")
    ap.add_argument("--coverage", type=float, default=0.297,
                    help="share of answer positions the gate keeps")
    ap.add_argument("--metric", default="acc",
                    choices=["acc", "acc_strict"],
                    help="acc is the lenient metric Table 9 of the paper uses; "
                         "acc_strict requires the answer format and is the "
                         "safer choice when the student may not have learned it")
    ap.add_argument("--plan", action="store_true",
                    help="write the configs and print the commands, run nothing")
    ap.add_argument("--decide", action="store_true",
                    help="skip the stages and read results already on disk")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-share", type=float, default=0.015,
                    help="smallest stratum, as a share of the pool")
    ap.add_argument("--k-max", type=int, default=50)
    ap.add_argument("--store-batch-size", type=int, default=8)
    ap.add_argument("--lora", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accumulate", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.decide:
        pairs = read_pairs(args.pairs)
        evalset = read_pairs(args.eval_pairs)
        print(f"{len(pairs)} training sequences, {len(evalset)} to score on")
        write_pairs(pairs, out / "corpus.json")
        write_pairs(evalset, out / "evalset.json")
        (out / "pool.json").write_text(
            json.dumps({"idx": list(range(len(pairs)))}))

        build = build_config(args, out, len(pairs))
        (out / "build_config.json").write_text(json.dumps(build, indent=1))
        (out / "train_config.json").write_text(
            json.dumps(train_config(args, out), indent=1))
        fs = build["experiments"][0]["fold_selection"]
        print(f"{args.folds} folds of {len(pairs) // args.folds}, "
              f"keeping {args.budget} each ({fs['fraction']:.1%}), "
              f"{2 * args.folds} fine-tunes")
        if args.plan:
            print("\nconfigs written; the commands are")

        run([sys.executable, str(CODE / "01_build_store" / "build_store.py"),
             "--config", str(out / "build_config.json"),
             "--stage", "store", "--only", "pilot"], args.plan)
        run([sys.executable, str(CODE / "01_build_store" / "build_store.py"),
             "--config", str(out / "build_config.json"),
             "--stage", "fold_selection", "--only", "pilot"], args.plan)
        run([sys.executable,
             str(CODE / "02_train_students" / "run_experiments.py"),
             "--config", str(out / "train_config.json"),
             "--group", "pilot"], args.plan)
        if args.plan:
            print("\nthen: python run_pilot.py --decide "
                  f"--out {out} --pairs {args.pairs} "
                  f"--eval-pairs {args.eval_pairs} --folds {args.folds}")
            return 0

    decide(args, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
