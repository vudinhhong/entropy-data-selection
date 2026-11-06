"""Turning the published artifact into something you can train on.

This is the piece a third party needs. Given the corpus (public), a list of
sequence indices (published alongside the store), and the store itself, it
produces a `torch.utils.data.Dataset` whose items carry the text *and* the
teacher's view of it, and a collate function that pads them into batches.

The one subtlety is alignment, and it is noted plainly because getting
it wrong is silent. Row j of a sequence's block in the store is the teacher's
distribution at the position that *predicts* answer token j. In a left-to-right
model that prediction is made at index j-1. So a block of n answer tokens maps
onto model positions prompt_length-1 .. prompt_length+n-2. Measured on 25,912
positions, the top-1 identity of row j matches answer token j 92.2% of the
time and token j+1 only 4.5% of the time, which is the check to run if you
suspect an off-by-one.

Typical use:

    from kdstore import LogitStore, LogitStoreDataset, collate_fn, load_pairs
    from kdstore.models import load_tokenizer

    store = LogitStore("artifacts/store4b.npz")
    pairs = load_pairs("gsm8k")["train"]
    tok = load_tokenizer("Qwen/Qwen1.5-1.8B")
    data = LogitStoreDataset(pairs, subset_indices, tok, store=store, k=5)
    loader = DataLoader(data, batch_size=8, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tok.pad_token_id, 5))
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .models import MAX_LENGTH, encode


class LogitStoreDataset(Dataset):
    """Sequences paired with the teacher's stored top-K view of them.

    Each item is (input_ids, prompt_length, topk_ids, topk_probs, entropy).
    With `store=None` the last three are None and the dataset is a plain
    text-only fine-tuning set, that is how the baseline arm is built, from
    the same code, so the two arms differ in nothing else.
    """

    def __init__(self, pairs, indices, tokenizer, store=None, k=0,
                 max_length=MAX_LENGTH, gate_source="entropy", gate_seed=1234):
        """gate_source="random" replaces the entropy channel with seeded
        uniform draws in [0, 1).

        That gives a gate which keeps a chosen fraction of positions without
        looking at the teacher at all: set the threshold to 1 - rho and the
        arm supervises a random rho of positions, matched in count to the
        entropy gate but not in content. It is the control that separates
        "uncertain positions carry the signal" from "any rho of positions
        would do", and it needs no change to the training loop, the gate is
        still a threshold on a per-position scalar.
        """
        self.items = []
        use_store = store is not None and k > 0
        rng = np.random.default_rng(gate_seed) if gate_source == "random" else None

        for i in indices:
            question, answer = pairs[i]
            ids, prompt_length = encode(tokenizer, question, answer, max_length)
            n_answer = len(ids) - prompt_length

            if not use_store:
                self.items.append((ids, prompt_length, None, None, None))
                continue

            start, stop = store.span(i)
            if stop - start != n_answer:
                raise ValueError(
                    f"sequence {i}: store holds {stop - start} positions but "
                    f"the tokenizer produced {n_answer} answer tokens. The "
                    f"store was built with a different tokenizer or a "
                    f"different max_length.")

            rows = np.arange(start, stop)
            ids_k, probs_k = store.view(rows, k)
            if rng is None:
                channel = store.entropy(rows).astype(np.float32)
            else:
                channel = rng.random(len(rows)).astype(np.float32)
            self.items.append((ids, prompt_length, ids_k, probs_k, channel))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_fn(batch, pad_token_id, k):
    """Pad a batch and lay the teacher's rows onto model positions.

    Returns (input_ids, attention_mask, labels, topk_ids, topk_probs, entropy).
    Labels are -100 on the prompt, so loss is taken on the answer only.
    Entropy is +inf outside the answer, which keeps those positions out of any
    gate without needing a second mask.
    """
    length = max(len(item[0]) for item in batch)
    size = len(batch)
    has_store = batch[0][2] is not None

    input_ids = torch.full((size, length), pad_token_id, dtype=torch.long)
    attention = torch.zeros((size, length), dtype=torch.long)
    labels = torch.full((size, length), -100, dtype=torch.long)
    entropy = torch.full((size, length), float("inf"), dtype=torch.float32)
    topk_ids = torch.zeros((size, length, k), dtype=torch.long) if has_store else None
    topk_probs = torch.zeros((size, length, k), dtype=torch.float32) if has_store else None

    for row, (ids, prompt_length, ids_k, probs_k, H) in enumerate(batch):
        n = len(ids)
        input_ids[row, :n] = torch.tensor(ids)
        attention[row, :n] = 1
        labels[row, prompt_length:n] = torch.tensor(ids[prompt_length:])
        if has_store:
            # answer token j is predicted at index j-1
            lo, hi = prompt_length - 1, n - 1
            topk_ids[row, lo:hi] = torch.from_numpy(ids_k)
            topk_probs[row, lo:hi] = torch.from_numpy(probs_k)
            entropy[row, lo:hi] = torch.from_numpy(H)

    return input_ids, attention, labels, topk_ids, topk_probs, entropy


def build_dataloader(pairs, indices, tokenizer, store=None, k=0, batch_size=8,
                     shuffle=True, max_length=MAX_LENGTH,
                     gate_source="entropy", gate_seed=1234, **kwargs):
    """Dataset and loader in one call, with the collate wired up."""
    data = LogitStoreDataset(pairs, indices, tokenizer, store=store, k=k,
                             max_length=max_length, gate_source=gate_source,
                             gate_seed=gate_seed)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, k), **kwargs)
    return data, loader


def gate_mask(entropy, tau, direction="high"):
    """The entropy gate, on a batch of positions.

    `high` keeps H > tau. That is where the teacher's distribution carries
    information the gold token does not: above one nat the top-1 probability
    averages about 0.50, so the distribution is genuinely multi-modal. Below
    0.01 nats the top-1 probability is 0.9998 and is the gold token, so the
    logits add nothing to plain cross-entropy, which is what `low` isolates.
    """
    if direction == "all":
        return torch.ones_like(entropy, dtype=torch.bool)
    return entropy < tau if direction == "low" else entropy > tau
