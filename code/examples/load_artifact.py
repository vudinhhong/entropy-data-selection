"""The shortest path from a downloaded store to a training batch.

Run it with no arguments and it inspects the store, which needs nothing but
numpy. Add --build-batch and it will also tokenize a subset and assemble one
training batch, which needs `transformers` and `datasets` and will download the
corpus and a tokenizer on first use.

    python examples/load_artifact.py
    python examples/load_artifact.py --build-batch --k 5

What the file gives you, per position: the teacher's K largest next-token
probabilities, the identities they belong to, and the entropy of the full
distribution. What it does not give you is the shape of the tail, only its
total mass, as 1 minus the sum of the K you have. Any loss you write against
this file has to treat the tail as one bucket. `kdstore.censored_kd_loss` does
that; see the note there on why renormalising the K entries instead is a bad
idea.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdstore import LogitStore, binary_entropy
from kdstore.paths import resolve

DEFAULT_STORE = "artifacts/store4b.npz"
DEFAULT_SUBSET = "artifacts/subsets/v2_subset_s11.json"


def inspect(store, tau):
    print(f"\n{store}\n")
    summary = store.describe(tau=tau)
    for key, value in summary.items():
        if key == "path":
            continue
        print(f"  {key:<28s} {value}")

    print(f"\n  The gate at tau = {tau} keeps {summary['rho'] * 100:.1f}% of "
          f"positions.")
    print(f"  Naming which ones costs h(rho) = "
          f"{binary_entropy(summary['rho']):.3f} bits per position, or "
          f"{store.n_positions * binary_entropy(summary['rho']) / 8 / 1e3:.1f} KB "
          f"for the whole store.")

    # What one position actually looks like.
    H = store.entropy()
    uncertain = int(np.argmax(H))
    ids, probs = store.view(np.array([uncertain]), min(5, store.k_max))
    print(f"\n  The most uncertain position in the file (row {uncertain}, "
          f"H = {H[uncertain]:.4f} nats):")
    for rank, (token_id, p) in enumerate(zip(ids[0], probs[0]), start=1):
        print(f"    {rank}. token {int(token_id):>7d}   p = {p:.4f}")
    print(f"    tail (everything else), total mass "
          f"{1 - probs[0].sum():.4f}")

    confident = int(np.argmin(H))
    ids, probs = store.view(np.array([confident]), 1)
    print(f"\n  The most certain position (row {confident}, "
          f"H = {H[confident]:.6f} nats): token {int(ids[0][0])} at "
          f"p = {probs[0][0]:.6f}")
    print("  This is the kind of position the gate drops: the teacher is sure, "
          "so\n  its distribution says nothing that the gold token does not.")


def build_batch(store, subset_path, model_id, k, tau):
    import torch

    from kdstore import (LogitStoreDataset, collate_fn, gate_mask, load_pairs,
                         censored_kd_loss)
    from kdstore.models import load_tokenizer

    indices = json.loads(Path(subset_path).read_text())["idx"]
    print(f"\n  subset: {len(indices)} sequences from {Path(subset_path).name}")

    pairs = load_pairs("gsm8k")["train"]
    tokenizer = load_tokenizer(model_id)
    data = LogitStoreDataset(pairs, indices[:8], tokenizer, store=store, k=k)
    batch = collate_fn([data[i] for i in range(4)], tokenizer.pad_token_id, k)
    input_ids, attention, labels, topk_ids, topk_probs, entropy = batch

    print(f"  batch of 4: input_ids {tuple(input_ids.shape)}, "
          f"topk_ids {tuple(topk_ids.shape)}")

    answer = labels != -100
    source = torch.zeros_like(answer)
    source[:, :-1] = answer[:, 1:]
    kept = gate_mask(entropy[source], tau, "high")
    print(f"  answer positions: {int(source.sum())}, "
          f"of which {int(kept.sum())} pass the gate "
          f"({float(kept.float().mean()) * 100:.1f}%)")

    # A loss against a stand-in student, so the shapes are exercised end to end.
    vocab = int(topk_ids.max()) + 1
    logits = torch.randn(int(source.sum()), vocab)
    total, kd, n = censored_kd_loss(
        logits, labels[answer], topk_ids[source], topk_probs[source], kept)
    print(f"  censored KD loss over {n} positions: "
          f"total {float(total):.1f}, KD term {float(kd):.1f} "
          f"(random student, so these are large by construction)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--tau", type=float, default=0.0335)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="Qwen/Qwen1.5-1.8B",
                    help="only used to pick a tokenizer")
    ap.add_argument("--build-batch", action="store_true",
                    help="also assemble a training batch (needs transformers)")
    args = ap.parse_args()

    store = LogitStore(resolve(args.store))
    inspect(store, args.tau)

    if args.build_batch:
        try:
            build_batch(store, resolve(args.subset), args.model, args.k,
                        args.tau)
        except ImportError as exc:
            print(f"\n  skipped batch building: {exc}")
            print("  install transformers and datasets to run this part")
    else:
        print("\n  pass --build-batch to assemble a training batch as well")


if __name__ == "__main__":
    main()
