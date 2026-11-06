"""Where does a corpus sit on the disagreement axis?

The sign of the selection criterion was read off how often the teacher own top
prediction differs from the reference token at the gated positions: 25.3% on
GSM8K where the uncertain tenth wins, 66.0% on ECQA where it loses. This runs
the same measurement on any corpus that has a fine-tuned teacher and a student
half, so the axis can be checked out of sample.

    python corpus_dis.py ape210k
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "code")
from kdstore import load_pairs
from kdstore.models import encode, load_model, load_tokenizer

TOKENIZER = "Qwen/Qwen3-4B-Base"
SAMPLE = 3000
COVERAGE = 0.297


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    corpus = sys.argv[1]
    teacher = f"runs/teacher_{corpus}"
    pool = f"artifacts/subsets/split_B_{corpus}.json"
    for pth in (teacher, pool):
        if not Path(pth).exists():
            sys.exit(f"missing {pth}")

    tok = load_tokenizer(TOKENIZER)
    model = load_model(teacher, train=False, device="cuda")
    model.eval()

    pairs = load_pairs(corpus)["train"]
    idx = json.loads(Path(pool).read_text())["idx"]
    rng = np.random.default_rng(0)
    take = rng.choice(len(idx), size=min(SAMPLE, len(idx)), replace=False)
    chosen = [idx[int(i)] for i in take]
    print(f"{corpus}: scoring {len(chosen)} sequences of the {len(idx):,}-"
          f"sequence student half", flush=True)

    ent, dis = [], []
    with torch.no_grad():
        for n, i in enumerate(chosen):
            ids, plen = encode(tok, *pairs[i])
            if len(ids) - plen < 2:
                continue
            t = torch.tensor([ids], device="cuda")
            logits = model(t).logits[0].float()
            lp = torch.log_softmax(logits[plen - 1:-1], -1)
            p = lp.exp()
            ent.append((-(p * lp).sum(-1)).cpu().numpy())
            dis.append((lp.argmax(-1) != t[0, plen:]).cpu().numpy())
            if (n + 1) % 500 == 0:
                print(f"  {n+1}/{len(chosen)}", flush=True)

    H = np.concatenate(ent)
    D = np.concatenate(dis)
    tau = float(np.quantile(H, 1.0 - COVERAGE))
    gate = H > tau
    rate = float(D[gate].mean())
    print(f"\n  answer positions      : {len(H):,}")
    print(f"  tau at {100*COVERAGE:.1f}% coverage : {tau:.4f} nats "
          f"(actual {100*gate.mean():.1f}%)")
    print(f"  mean entropy          : {H.mean():.3f} nats")
    print(f"  TEACHER TOP-1 DIFFERS FROM THE REFERENCE: {100*rate:.1f}%")
    print(f"\n  GSM8K 25.3% (high wins) | MATH 44.5% | "
          f"ECQA 66.0% (low wins) | Orca 19.5% (high LOSES)")
    out = (f"results/disagreement_{corpus}_"
           f"{teacher.replace(chr(47), chr(45))}_n{len(ent)}_cov30.json")
    Path("results").mkdir(exist_ok=True)
    Path(out).write_text(json.dumps({
        "corpus": corpus, "teacher": teacher, "sequences": len(ent),
        "positions": int(len(H)), "tau": tau, "coverage": float(gate.mean()),
        "mean_entropy": float(H.mean()), "disagreement_at_gated": rate,
    }, indent=2))
    print(f"  written {out}")


if __name__ == "__main__":
    main()
