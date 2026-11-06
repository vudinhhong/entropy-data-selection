"""Is the criterion tracking what one particular student finds hard?

The criterion ranks sequences by the teacher's entropy summed over the gated
positions. Instruction-following difficulty (IFD) ranks them by how much the
question helps with the answer, scored by the student itself. If the two were
measuring the same thing, the criterion's gain on one student would be a fact
about that student instead of about the teacher.

This prints the correlation between the two, for every student we have an IFD
score for, and the correlations of those difficulty scores with each other --
which turn out to be the more informative half of the table.

The gated score is recomputed from the store instead of read from a cache, so
a run of this script also checks that the gate reproduces the released
subsets.

    python code/12_criterion_vs_difficulty/correlate.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location(
    "kdstats", ROOT / "code" / "kdstore" / "stats.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

TAU = 0.0335                      # the gate the GSM8K folds were selected with
STORE = ROOT / "artifacts" / "store4b.npz"
META = ROOT / "artifacts" / "subsets" / "store4b.meta.json"
POOL = ROOT / "artifacts" / "subsets" / "split_B.json"
CHECK = ROOT / "artifacts" / "subsets" / "fold0_gated_H.json"
SUBSETS = ROOT / "artifacts" / "subsets"

STUDENTS = [
    ("Qwen1.5-1.8B", "ifd_gsm8k.json"),
    ("Llama-3.2-1B", "ifd_gsm8k_llama32_1b.json"),
    ("SmolLM2-360M", "ifd_gsm8k_smol2_360m.json"),
    ("Pythia-410M", "superfilter_gsm8k.json"),
]


def main():
    meta = {r["i"]: r for r in json.loads(META.read_text())["meta"]}
    pool = json.loads(POOL.read_text())["idx"]
    with np.load(STORE) as z:
        offsets, H = z["offsets"], z["H"].astype(np.float64)

    def gated(i):
        h = H[offsets[i]:offsets[i + 1]]
        return float(h[h > TAU].sum())

    # The gate has to be the one the paper selected with, or the correlations
    # below describe a quantity nothing was trained on.
    released = json.loads(CHECK.read_text())
    want = released["info"]["summary"]["gated_positions"]
    got = sum(int((H[offsets[i]:offsets[i + 1]] > TAU).sum())
              for i in released["idx"])
    if got != want:
        raise SystemExit(f"gate does not reproduce the released subset: "
                         f"{got} gated positions against {want}")
    print(f"gate reproduces fold 0's released subset ({want} positions)\n")

    scores, missing = {}, []
    for label, name in STUDENTS:
        path = SUBSETS / name
        if not path.exists():
            missing.append(label)
            continue
        d = json.loads(path.read_text())
        scores[label] = ({int(k): v["ifd"] for k, v in d["scores"].items()},
                         d["model"])
    if missing:
        print(f"no IFD file for {', '.join(missing)}; "
              f"run code/09_selection_baselines/score_ifd.py for them\n")

    labels = [l for l, _ in STUDENTS if l in scores]
    keys = [i for i in pool if meta[i].get("npos", 0) > 0
            and all(i in scores[l][0] for l in labels)]
    print(f"{len(keys)} sequences scored by all {len(labels)} students\n")

    g = [gated(i) for i in keys]
    series = {l: [scores[l][0][i] for i in keys] for l in labels}

    print("the teacher's gated entropy against each student's IFD")
    for l in labels:
        r = S.pearson(g, series[l])
        print(f"  {l:14s} {scores[l][1]:28s} r = {r['r']:+.3f}  "
              f"p = {r['p']:.1e}")

    print("\nthe difficulty scores against each other")
    print(" " * 16 + "".join(f"{l:>15s}" for l in labels))
    for a in labels:
        row = f"  {a:14s}"
        for b in labels:
            row += ("            ---" if a == b else
                    f"{S.pearson(series[a], series[b])['r']:>+15.3f}")
        print(row)


if __name__ == "__main__":
    main()
