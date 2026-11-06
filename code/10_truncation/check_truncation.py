"""Does the selection score survive seeing only the top K logprobs?

The criterion needs a per-position entropy, which is a sum over the whole
vocabulary. A practitioner reading an inference API sees only the top few
logprobs. This script asks what that costs, by reconstructing each position's
entropy from the top K stored probabilities with the residual mass spread
uniformly over the omitted tokens, and then re-ranking the sequences.

It needs no GPU and no model: the store already carries the exact entropy per
position, computed from the full distribution when it was dumped, beside the
top-50 view. Everything below is numpy over those two arrays.

    python check_truncation.py

Reproduces the three columns of the truncation paragraph in the paper.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "artifacts" / "store4b.npz"
SPLIT = ROOT / "artifacts" / "subsets" / "split_B.json"
VOCAB = 151_936
TAU = 0.0335          # the GSM8K gate, in nats
WIDTHS = (5, 10, 20, 50)
SAMPLE = 200_000      # positions used for the entropy-error column


def reconstruct(p, k):
    """Entropy from the top k probabilities, residual mass spread uniformly.

    The omitted tokens are given r/(V-k) each, so their contribution to the
    entropy is r*log((V-k)/r). This is the most an observer can say who sees
    k probabilities and knows they do not sum to one.
    """
    top = p[:, :k]
    residual = np.clip(1.0 - top.sum(1), 1e-12, None)
    kept = -(top * np.log(np.clip(top, 1e-12, None))).sum(1)
    return kept + residual * np.log((VOCAB - k) / residual)


def sequence_scores(entropy, offsets, keep):
    """g(s): the entropy a sequence carries at its gated positions."""
    out = np.empty(len(keep))
    for j, i in enumerate(keep):
        h = entropy[offsets[i]:offsets[i + 1]]
        out[j] = h[h > TAU].sum()
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main():
    store = np.load(STORE, allow_pickle=True)
    logp = store["logp"].astype(np.float32)
    exact = store["H"].astype(np.float32)
    offsets = store["offsets"]
    half = np.array(sorted(json.loads(SPLIT.read_text())["idx"]))
    probs = np.exp(logp)

    sample = np.random.default_rng(0).choice(len(logp), SAMPLE, replace=False)
    reference = sequence_scores(exact, offsets, half)
    decile = len(half) // 10
    chosen = set(np.argsort(-reference)[:decile].tolist())

    print(f"{len(logp):,} positions, {len(half):,} sequences in the student "
          f"half, top decile = {decile}")
    print(f"\n{'K':>3} {'mean |dH| (nats)':>18} {'Spearman':>10} "
          f"{'same tenth':>12}")
    for k in WIDTHS:
        approx = reconstruct(probs, k)
        scores = sequence_scores(approx, offsets, half)
        overlap = set(np.argsort(-scores)[:decile].tolist()) & chosen
        print(f"{k:>3} {np.abs(approx[sample] - exact[sample]).mean():18.4f} "
              f"{spearman(reference, scores):10.4f} "
              f"{100 * len(overlap) / decile:11.1f}%")


if __name__ == "__main__":
    main()
