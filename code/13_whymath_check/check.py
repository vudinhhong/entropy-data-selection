"""Recompute the corpus comparison table from the three stores.

Every row of it except the teacher's end-to-end accuracy is a function of the
store alone: the gold token's rank gives the disagreement rate and the top-five
miss rate, the gold token's probability gives the last row, and the entropy
array gives the gate and the score. Nothing here needs a GPU or a teacher
checkpoint, so the table can be checked at any time.

    python code/13_whymath_check/check.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPORA = [
    # name, store, meta, pool, tau as the config states it
    ("GSM8K", "store4b.npz", "subsets/store4b.meta.json",
     "split_B.json", 0.0335),
    ("MATH", "store4b_math.npz", "subsets/store4b_math.meta.json",
     "split_B_math.json", 0.4404),
    ("ECQA", "store4b_ecqa.npz", "store4b_ecqa.meta.json",
     "split_B_ecqa.json", "coverage:0.297"),
]
PAPER = {  # the values the paper prints, for comparison
    "GSM8K": dict(tau=0.0335, meanH=0.131, indep=12.8, dis=25.3,
                  miss=3.1, goldp=0.697),
    "MATH": dict(tau=0.4404, meanH=0.405, indep=3.4, dis=44.5,
                 miss=10.0, goldp=0.429),
    "ECQA": dict(tau=0.9111, meanH=0.627, indep=4.2, dis=66.0,
                 miss=26.8, goldp=0.247),
}


def pearson_r(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


print(f"{'':36s}{'computed':>12s}{'paper':>10s}")
for name, store, meta, pool, tau_spec in CORPORA:
    sp = ROOT / "artifacts" / store
    mp = ROOT / "artifacts" / meta
    pp = ROOT / "artifacts" / "subsets" / pool
    if not (sp.exists() and mp.exists() and pp.exists()):
        print(f"\n{name}: missing inputs, skipped")
        continue
    with np.load(sp) as z:
        offsets = z["offsets"]
        H = z["H"].astype(np.float64)
        gold_rank = z["gold_rank"]
        gold_p = z["gold_p"].astype(np.float64)
    records = json.loads(mp.read_text())["meta"]
    allowed = set(json.loads(pp.read_text())["idx"])

    keep = np.zeros(len(H), bool)
    for r in records:
        if r["i"] in allowed and r.get("npos", 0) > 0:
            keep[offsets[r["i"]]:offsets[r["i"] + 1]] = True

    if isinstance(tau_spec, str):
        want = float(tau_spec.split(":", 1)[1])
        tau = float(np.quantile(H[keep], 1.0 - want))
    else:
        tau = float(tau_spec)
    gate = keep & (H > tau)

    gated_H, gated_n = [], []
    for r in records:
        if r["i"] not in allowed or r.get("npos", 0) <= 0:
            continue
        lo, hi = offsets[r["i"]], offsets[r["i"] + 1]
        h = H[lo:hi]
        passed = h > tau
        gated_H.append(h[passed].sum())
        gated_n.append(int(passed.sum()))
    r2 = pearson_r(gated_H, gated_n) ** 2

    p = PAPER[name]
    rows = [
        ("Gate threshold tau (nats)", tau, p["tau"], 4),
        ("Mean entropy over answer positions", H[keep].mean(), p["meanH"], 3),
        ("Coverage the gate reaches (%)", 100 * gate.sum() / keep.sum(),
         None, 1),
        ("Independent variance of the score (%)", 100 * (1 - r2),
         p["indep"], 1),
        ("Its top-1 differs from the reference (%)",
         100 * (gold_rank[gate] > 0).mean(), p["dis"], 1),
        ("Reference outside its top five (%)",
         100 * (gold_rank[gate] >= 5).mean(), p["miss"], 1),
        ("Mean probability on the reference token",
         gold_p[gate].mean(), p["goldp"], 3),
    ]
    print(f"\n{name}  ({keep.sum():,} answer positions in the student half, "
          f"{gate.sum():,} gated)")
    for label, got, want_v, places in rows:
        flag = ""
        if want_v is not None:
            flag = ("  MATCH" if abs(got - want_v) < 0.5 * 10 ** -places * 11
                    else "  <-- DIFFERS")
        shown = "" if want_v is None else f"{want_v:>10.{places}f}"
        print(f"  {label:34s}{got:>12.{places}f}{shown}{flag}")


# --------------------------------------------------------------- teacher row

# The one row above that is not a function of the store is the teacher's
# end-to-end accuracy on the prompts the students train on. It is computed
# here from the teacher's saved generations, which are released, instead of
# from a teacher checkpoint, which is not.
print("\n\nTeacher's end-to-end accuracy on the prompts the students train on")
for name, directory, want in (("GSM8K", "teacher_solutions", 85.5),
                              ("MATH", "teacher_solutions_math", 50.7)):
    files = sorted((ROOT / "artifacts" / directory).glob("*.json"))
    if not files:
        print(f"  {name}: no saved generations")
        continue
    rates, n = [], 0
    for f in files:
        d = json.loads(f.read_text())
        rates.append(100 * d["accuracy"])
        n = len(d["pairs"])
    mean = sum(rates) / len(rates)
    flag = "MATCH" if abs(mean - want) < 0.05 else f"paper says {want}"
    print(f"  {name:6s} {len(rates)} subsets of {n} prompts: "
          + ", ".join(f"{r:.2f}" for r in rates)
          + f"  mean {mean:.2f}   paper {want}  {flag}")
print("  ECQA   measured with code/09_selection_baselines/measure_teacher.py; "
      "its teacher is in runs/teacher_ecqa")
