"""Instruction-following difficulty (IFD) of Li et al.

IFD is the ratio of a model's loss on the answer given the question to its
loss on the answer alone.

    IFD(Q, A) = L(A | Q) / L(A)

A ratio near one means the question did not help, which is read as the sample
carrying information the model does not already have. Li et al. keep the
highest scores below one. Superfiltering is the same ratio computed with a
small proxy model instead of the one being tuned, so the two differ only in
`--model` and both are produced here.

One departure from the paper, stated because it matters: Li et al. score with
a "pre-experienced" model briefly trained on a sample of the data. We score
with the untrained base. That makes this a faithful implementation of the
score and an approximation of the full recipe.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from kdstore import Report, load_config, load_pairs          # noqa: E402
from kdstore.models import (MAX_LENGTH, load_model,          # noqa: E402
                            load_tokenizer)
from kdstore.paths import prepare, resolve                   # noqa: E402


@torch.no_grad()
def answer_loss(model, tok, prompt, answer, device):
    """Mean cross-entropy on the answer tokens, with and without the prompt."""
    out = {}
    for name, text, prefix in (("conditioned", prompt + answer, prompt),
                               ("direct", answer, "")):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=MAX_LENGTH).input_ids.to(device)
        n_prefix = len(tok(prefix).input_ids) if prefix else 0
        if ids.shape[1] - n_prefix < 2:
            out[name] = None
            continue
        labels = ids.clone()
        labels[:, :n_prefix] = -100
        out[name] = float(model(ids, labels=labels).loss)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--only", help="run one named experiment")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = Report("Selection baselines: instruction-following difficulty")

    for spec in cfg["experiments"]:
        if args.only and spec["name"] != args.only:
            continue
        pairs = load_pairs(spec["dataset"])["train"]
        pool = json.loads(resolve(spec["pool"]).read_text())["idx"]
        device = spec.get("device", "cuda")
        tok = load_tokenizer(spec["model"])
        model = load_model(spec["model"], train=False, device=device)
        model.eval()

        scores, skipped = {}, 0
        for n, i in enumerate(pool):
            prompt, answer = pairs[i]
            loss = answer_loss(model, tok, prompt, answer, device)
            if loss["conditioned"] is None or not loss["direct"]:
                skipped += 1
                continue
            scores[int(i)] = {"conditioned": loss["conditioned"],
                              "direct": loss["direct"],
                              "ifd": loss["conditioned"] / loss["direct"]}
            if n % 500 == 0:
                print(f"  {spec['name']}: {n}/{len(pool)}", flush=True)

        path = prepare(resolve(spec["output"]))
        path.write_text(json.dumps({"model": spec["model"],
                                    "dataset": spec["dataset"],
                                    "scores": scores}))
        values = [v["ifd"] for v in scores.values()]
        values.sort()
        report.table(spec["name"],
                     ["scored", "skipped", "min", "median", "max", "share < 1"],
                     [[len(values), skipped, round(values[0], 3),
                       round(values[len(values) // 2], 3), round(values[-1], 3),
                       f"{100 * sum(1 for v in values if v < 1) / len(values):.1f}%"]],
                     title=f"IFD with {spec['model']}")
        del model
        torch.cuda.empty_cache()

    out = resolve(cfg.get("report_dir", "results"))
    stem = "selection_baselines"
    if args.only:
        stem += "_" + args.only
    else:
        stem += "_" + "_".join(e["name"] for e in cfg["experiments"])
    report.save(out / f"{stem}.json", out / f"{stem}.txt")


if __name__ == "__main__":
    main()
