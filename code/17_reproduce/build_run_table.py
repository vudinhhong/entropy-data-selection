"""Rebuild data/per_run_accuracies.csv from raw run output.

Each training run writes one JSON into a results directory, holding its
settings, its evaluation size and its accuracy, along with the per-item
generations, which are far too large to release and are dropped here. This
script walks one or more such directories and writes the per-run table the
paper's statistics are computed from.

    python code/17_reproduce/build_run_table.py results/grid
    python code/17_reproduce/build_run_table.py results/grid results/extra_*

Runs are keyed by tag; the first directory on the command line wins a clash,
so the authoritative source goes first.
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Works inside this repository and also when the file is copied somewhere
# else on its own, on a cluster, for instance, to strip the generations out
# of a results directory before anything is transferred.
OUT = (ROOT / "data" / "per_run_accuracies.csv" if (ROOT / "data").is_dir()
       else Path.cwd() / "per_run_accuracies.csv")
COLS = ["tag", "group", "source_dir", "corpus", "student", "rule", "fold",
        "seed", "n_train", "k", "tau", "gate", "lam", "temperature", "lr",
        "epochs", "lora_r", "subset", "n_eval", "acc_lenient", "acc_strict",
        "se"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="results directories to walk")
    ap.add_argument("-o", "--out", default=str(OUT))
    args = ap.parse_args()

    rows, seen, clashes = [], set(), 0
    for d in args.dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                j = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            tag = j.get("tag")
            if not tag or "acc" not in j:
                continue
            if tag in seen:
                clashes += 1
                continue
            seen.add(tag)
            st, ti = j.get("settings") or {}, j.get("train_info") or {}
            m = re.search(r"_f(\d+)_(.+?)_(\d+)$", tag)
            rows.append({
                "tag": tag, "group": j.get("group", ""), "source_dir": d,
                "corpus": st.get("dataset", ""), "student": st.get("model", ""),
                "rule": m.group(2) if m else "", "fold": m.group(1) if m else "",
                "seed": m.group(3) if m else "", "n_train": ti.get("n_train", ""),
                "k": st.get("k", ""), "tau": st.get("tau", ""),
                "gate": st.get("gate", ""), "lam": st.get("lam", ""),
                "temperature": st.get("temperature", ""), "lr": st.get("lr", ""),
                "epochs": st.get("epochs", ""), "lora_r": st.get("lora", ""),
                "subset": st.get("subset", ""), "n_eval": j.get("n", ""),
                "acc_lenient": j.get("acc", ""),
                "acc_strict": j.get("acc_strict", ""), "se": j.get("se", "")})

    if not rows:
        sys.exit("no runs found; check the directories given")
    rows.sort(key=lambda r: (r["group"], r["corpus"], r["rule"], r["fold"],
                             r["seed"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} runs, "
          f"{len({r['group'] for r in rows})} groups, "
          f"{len({r['corpus'] for r in rows if r['corpus']})} corpora")
    if clashes:
        print(f"{clashes} duplicate tags skipped; the earliest directory won")
    print("check it with: python code/17_reproduce/verify_paper_numbers.py")


if __name__ == "__main__":
    main()
