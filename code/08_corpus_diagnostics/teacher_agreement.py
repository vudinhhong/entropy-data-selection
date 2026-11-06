"""How often does the teacher disagree with the reference, at gated positions?

This is the quantity the sign of the selection criterion tracks, and the whole
point of it is that it costs one teacher-forced pass and no training. The pass
is already in the store: `gold_rank` is where the reference token sat in the
teacher's ranking at that position, so the disagreement rate is a count over
an array.

Covers all three corpora; compare_corpora.py covers GSM8K and MATH only.

    python teacher_agreement.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# corpus, store, student-half split, gate threshold in nats. Each threshold is
# set per corpus so that the gate passes about the same share of positions.
CORPORA = [
    ("GSM8K", "store4b.npz", "split_B.json", 0.0335),
    ("MATH", "store4b_math.npz", "split_B_math.json", 0.4404),
    ("ECQA", "store4b_ecqa.npz", "split_B_ecqa.json", 0.9111),
]


def main():
    head = (f"{'corpus':8s} {'tau':>7s} {'mean H':>8s} {'coverage':>9s} "
            f"{'top-1 != ref':>13s} {'ref outside top 5':>18s} {'p(ref)':>8s}")
    print(head)
    print("-" * len(head))
    for name, store_file, split_file, tau in CORPORA:
        store = np.load(ROOT / "artifacts" / store_file, allow_pickle=True)
        entropy = store["H"].astype(np.float32)
        gold_rank = store["gold_rank"]
        gold_p = store["gold_p"].astype(np.float32)
        offsets = store["offsets"]

        half = sorted(json.loads(
            (ROOT / "artifacts" / "subsets" / split_file).read_text())["idx"])
        answer = np.zeros(len(entropy), dtype=bool)
        for i in half:
            answer[offsets[i]:offsets[i + 1]] = True
        gated = answer & (entropy > tau)

        print(f"{name:8s} {tau:7.4f} {entropy[answer].mean():8.3f} "
              f"{100 * gated.sum() / answer.sum():8.1f}% "
              f"{100 * (gold_rank[gated] > 0).mean():12.1f}% "
              f"{100 * (gold_rank[gated] >= 5).mean():17.1f}% "
              f"{gold_p[gated].mean():8.3f}")


if __name__ == "__main__":
    main()
