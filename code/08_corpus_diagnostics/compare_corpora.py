"""Why the selection criterion carries over to one corpus and not the other.

The criterion is the same code on both corpora, the gate is set to the same
coverage on both, and it works on GSM8K and not on MATH. This script asks the
data what differs, instead of reasoning about it from the outside. It reports
three things per corpus: how the first layer's difficulty proxy is
distributed, how much of the score is anything other than a count of gated
positions, and what kind of token the gate actually fires on.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from kdstore import annotate_gate                      # noqa: E402
from kdstore.store import LogitStore                   # noqa: E402

CORPORA = [
    ("GSM8K", "store4b.npz", "store4b.meta.json", "split_B.json", 0.0335),
    ("MATH", "store4b_math.npz", "store4b_math.meta.json",
     "split_B_math.json", 0.4404),
]

LATEX = set("\\${}^_&")


def token_kind(text):
    """A coarse bucket, enough to tell notation from arithmetic from prose."""
    stripped = text.strip()
    if stripped == "":
        return "whitespace"
    if any(ch in stripped for ch in LATEX):
        return "LaTeX markup"
    if re.fullmatch(r"[0-9][0-9.,]*", stripped):
        return "number"
    if re.fullmatch(r"[+\-*/=<>()\[\]|%]+", stripped):
        return "operator"
    if re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", stripped):
        return "word"
    return "punctuation/other"


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    rng = np.random.default_rng(0)

    for name, store_f, meta_f, split_f, tau in CORPORA:
        store = LogitStore(ROOT / "artifacts" / store_f)
        meta = json.loads((ROOT / "artifacts/subsets" / meta_f).read_text())["meta"]
        annotate_gate(meta, store, tau)
        pool = set(json.loads((ROOT / "artifacts/subsets" / split_f).read_text())["idx"])
        rows = [r for r in meta if r["i"] in pool and r.get("npos", 0) > 0]

        steps = np.array([r["steps"] for r in rows])
        gated_h = np.array([r["gated_H"] for r in rows])
        gated_n = np.array([r["gated"] for r in rows])

        print(f"\n{'=' * 64}\n{name}   tau = {tau}   {len(rows)} sequences\n{'=' * 64}")

        counts = collections.Counter(steps.tolist())
        biggest = max(counts.values())
        print("layer 1, the difficulty proxy (newlines in the answer)")
        print(f"  median {np.median(steps):.0f}, mean {steps.mean():.2f}, "
              f"max {steps.max()}")
        print(f"  largest single value holds {100 * biggest / len(rows):.1f}% "
              f"of the pool; {100 * (steps <= 1).mean():.1f}% have at most one")

        r = np.corrcoef(gated_h, gated_n)[0, 1]
        print("layer 2, what the score is, beyond a count of gated positions")
        print(f"  corr(score, gated count) = {r:+.3f}, so "
              f"{100 * (1 - r ** 2):.1f}% of its variance is something else")

        H = store.entropy()
        offsets = store.array("offsets")
        ids = store.array("idx")
        mask = np.zeros(len(H), dtype=bool)
        for row in rows:
            mask[offsets[row["i"]]:offsets[row["i"] + 1]] = True
        everywhere = rng.choice(np.flatnonzero(mask), 40000, replace=False)
        passed = rng.choice(np.flatnonzero(mask & (H > tau)), 40000, replace=False)

        def profile(where):
            kinds = collections.Counter(
                token_kind(tok.decode([int(t)])) for t in ids[where, 0])
            total = sum(kinds.values())
            return {k: 100 * v / total for k, v in kinds.items()}

        # Whether the teacher's uncertainty is about a text it agrees with.
        # gold_rank is where the reference token sits in the teacher's own
        # ranking, so rank > 0 means the teacher would have written something
        # else at that position.
        rank = store.array("gold_rank")
        prob = store.array("gold_p").astype(np.float32)
        print("the teacher, does its uncertainty concern text it agrees with?")
        for label, where in (("all positions", everywhere),
                             ("gated positions", passed)):
            print(f"  {label:18s} top-1 differs from the reference "
                  f"{100 * (rank[where] > 0).mean():5.1f}%   "
                  f"reference outside top-5 {100 * (rank[where] >= 5).mean():5.1f}%   "
                  f"mean P(reference) {prob[where].mean():.3f}")

        base, gated = profile(everywhere), profile(passed)
        print("the gate, what kind of token it fires on")
        print(f"  {'kind':18s} {'all':>7s} {'gated':>7s} {'enrichment':>11s}")
        for kind in sorted(set(base) | set(gated), key=lambda k: -gated.get(k, 0)):
            b, g = base.get(kind, 0.0), gated.get(kind, 0.0)
            ratio = f"{g / b:.2f}x" if b > 0.05 else "--"
            print(f"  {kind:18s} {b:6.1f}% {g:6.1f}% {ratio:>11s}")


if __name__ == "__main__":
    main()
