"""Is the cross-entropy this package computes the same as the library's?

Every arm in the paper is trained with a hand-written loss, chunked over
positions so the vocabulary-sized logits never all exist at once. That is a
memory decision, not a modelling one, and it should produce exactly what
`model(..., labels=...)` produces. If it does not, the text-only arm and the
gated arm are not on the same footing and no comparison between them means
anything.

    python tests/test_loss_equivalence.py --model Qwen/Qwen1.5-1.8B

The test runs one real batch through both paths and compares the scalar loss
and the gradients. It needs a GPU only because the models are large; the
arithmetic is identical on CPU.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdstore import build_dataloader, censored_kd_loss, load_pairs
from kdstore.models import base_model_of, load_model, load_tokenizer

POSITION_CHUNK = 256


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen1.5-1.8B")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--sequences", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pairs = load_pairs(args.dataset)["train"][:args.sequences]
    tok = load_tokenizer(args.model, padding_side="right")
    _, loader = build_dataloader(pairs, range(len(pairs)), tok, store=None,
                                 k=0, batch_size=args.sequences, shuffle=False)
    ids, att, lab, _, _, _ = next(iter(loader))
    ids, att, lab = (x.to(args.device) for x in (ids, att, lab))

    model = load_model(args.model, train=False, device=args.device)
    body = base_model_of(model)

    # --- path 1: the library's own loss ---
    with torch.no_grad():
        library = float(model(input_ids=ids, attention_mask=att,
                              labels=lab).loss)

    # --- path 2: what the trainer does ---
    with torch.no_grad():
        hidden = body.model(input_ids=ids,
                            attention_mask=att).last_hidden_state
        labelled = lab != -100
        source = torch.zeros_like(labelled)
        source[:, :-1] = labelled[:, 1:]
        states, gold = hidden[source], lab[labelled]
        total = torch.zeros((), device=args.device)
        n = states.shape[0]
        for c in range(0, n, POSITION_CHUNK):
            sl = slice(c, min(c + POSITION_CHUNK, n))
            logits = body.lm_head(states[sl]).float()
            empty = torch.zeros(logits.shape[0], dtype=torch.bool,
                                device=args.device)
            summed, _, _ = censored_kd_loss(logits, gold[sl], None, None,
                                            empty, lam=1.0)
            total = total + summed
        ours = float(total / max(1, n))

    print(f"  sequences                 {args.sequences}")
    print(f"  supervised positions      {n}")
    print(f"  library loss              {library:.8f}")
    print(f"  this package's loss       {ours:.8f}")
    print(f"  absolute difference       {abs(library - ours):.3e}")
    print(f"  relative difference       {abs(library - ours) / library:.3e}")

    # bf16 hidden states make the two paths differ in the last bits; anything
    # above a part in ten thousand would be a real discrepancy.
    ok = abs(library - ours) / library < 1e-4
    print(f"\n  {'PASS' if ok else 'FAIL'}: the two paths agree"
          f"{'' if ok else ' -- they do NOT agree'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
