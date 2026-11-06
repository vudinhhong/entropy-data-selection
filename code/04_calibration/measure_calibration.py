"""How well calibrated is a base model, before anyone spends GPU time on it.

Everything here is teacher-forced on corpus text at answer positions only, so
it costs one forward pass per model and no training at all. The point is to ask
whether a cheap property of the base model predicts how much it gains from the
artifact, if it did, you could tell in advance which cells are worth running.

    python measure_calibration.py
    python measure_calibration.py --dataset math --only Qwen/Qwen2-1.5B

Four quantities per model:

    entropy       mean H of the next-token distribution, in nats
    confidence    mean top-1 probability
    accuracy      how often the top-1 token is the one that follows
    ECE           expected calibration error over ten confidence bins
    over-confidence  signed version of the same: positive means the model
                  claims more certainty than it earns

Read the correlations in the report with the sample size in mind: six cells is
six points, so a correlation needs to be very strong before it means anything.
Both series must come from the same corpus: a calibration number measured on
one corpus and a gain measured on another can correlate by accident.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdstore import Report, load_config, load_pairs
from kdstore.models import (MAX_LENGTH, PROMPT, TARGET, load_model,
                            load_tokenizer, log, vram_needed, wait_for_vram)
from kdstore.paths import resolve
from kdstore.stats import pearson

HERE = Path(__file__).resolve().parent
BINS = 10


@torch.no_grad()
def measure(model_id, pairs, pool, n_sequences, device, tokenizer_id=None):
    """One model, teacher-forced over answer positions."""
    tokenizer = load_tokenizer(tokenizer_id or model_id)
    model = load_model(model_id, train=False, device=device)

    bin_confidence = np.zeros(BINS)
    bin_correct = np.zeros(BINS)
    bin_count = np.zeros(BINS)
    entropy_sum = confidence_sum = correct_sum = count = 0.0

    for i in pool[:n_sequences]:
        question, answer = pairs[i]
        prompt = tokenizer(PROMPT.format(q=question),
                           add_special_tokens=False)["input_ids"]
        target = tokenizer(TARGET.format(a=answer),
                           add_special_tokens=False)["input_ids"]
        ids = (prompt + target)[:MAX_LENGTH]
        if len(ids) <= len(prompt) + 1:
            continue

        logits = model(input_ids=torch.tensor([ids], device=device)) \
            .logits[0].float()
        # score the answer only: position j predicts token j+1
        start = len(prompt) - 1
        logits = logits[start:-1]
        gold = torch.tensor(ids[start + 1:], device=device)

        logp = torch.log_softmax(logits, -1)
        prob = logp.exp()
        confidence, predicted = prob.max(-1)
        correct = (predicted == gold).float()

        entropy_sum += float((-(prob * logp).sum(-1)).sum())
        confidence_sum += float(confidence.sum())
        correct_sum += float(correct.sum())
        count += confidence.numel()

        c = confidence.cpu().numpy()
        k = correct.cpu().numpy()
        which = np.clip((c * BINS).astype(int), 0, BINS - 1)
        for b in range(BINS):
            sel = which == b
            if sel.any():
                bin_count[b] += sel.sum()
                bin_confidence[b] += c[sel].sum()
                bin_correct[b] += k[sel].sum()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nonempty = bin_count > 0
    total = bin_count.sum()
    ece = float((bin_count[nonempty] / total
                 * np.abs(bin_confidence[nonempty] / bin_count[nonempty]
                          - bin_correct[nonempty] / bin_count[nonempty])).sum())
    # signed, so the direction of the miscalibration survives
    over = float(((bin_confidence[nonempty] - bin_correct[nonempty])
                  / total).sum())
    return {"n": int(count), "entropy": entropy_sum / count,
            "confidence": confidence_sum / count,
            "accuracy": correct_sum / count, "ece": ece,
            "over_confidence": over}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--dataset", help="override the corpus named in the config")
    ap.add_argument("--only", help="measure a single model")
    args = ap.parse_args()

    config = load_config(args.config)
    defaults = config.get("defaults", {})
    report = Report("Base-model calibration")
    collected = {}

    for spec in config["corpora"]:
        spec = {**defaults, **spec}
        if args.dataset and spec["dataset"] != args.dataset:
            continue
        pairs = load_pairs(spec["dataset"])["train"]
        pool = json.loads(resolve(spec["pool"]).read_text())["idx"]
        models = [m for m in spec["models"]
                  if not args.only or m["model"] == args.only]

        rows = []
        for entry in models:
            log(f"calibration: {entry['model']} on {spec['dataset']}")
            if spec.get("device", "cuda") != "cpu":
                wait_for_vram(vram_needed(entry["model"]), log=log)
            stats = measure(entry["model"], pairs, pool,
                            spec.get("n_sequences", 120),
                            spec.get("device", "cuda"),
                            entry.get("tokenizer"))
            collected.setdefault(spec["dataset"], {})[entry["label"]] = stats
            rows.append([entry["label"], stats["n"], stats["entropy"],
                         stats["confidence"], stats["accuracy"],
                         stats["ece"], stats["over_confidence"]])

        report.table(f"calibration_{spec['dataset']}",
                     ["model", "positions", "entropy", "confidence",
                      "next-token acc", "ECE", "over-confidence"],
                     rows, title=f"{spec['dataset']}: teacher-forced on "
                                 f"{spec.get('n_sequences', 120)} sequences",
                     places=4)

    # Correlate against the measured gains, if the caller supplied them.
    gains = config.get("cell_gains")
    if gains:
        rows = []
        for quantity in ("accuracy", "ece", "over_confidence", "entropy"):
            xs, ys = [], []
            for cell, gain in gains.items():
                dataset, label = cell.split("/")
                stats = collected.get(dataset, {}).get(label)
                if stats:
                    xs.append(stats[quantity])
                    ys.append(gain)
            if len(xs) >= 3:
                r = pearson(xs, ys)
                rows.append([quantity, r["n"], r["r"], r["p"],
                             f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"])
        if rows:
            report.table("correlations",
                         ["quantity", "n", "Pearson r", "p", "95% CI"], rows,
                         title="Does a cheap property predict the gain?",
                         places=4)

    out = resolve(config.get("report_dir", "results"))
    stem = "calibration"
    if args.dataset:
        stem += "_" + args.dataset
    if args.only:
        stem += "_" + args.only.replace("/", "-")
    report.save(out / f"{stem}.json", out / f"{stem}.txt")


if __name__ == "__main__":
    main()
