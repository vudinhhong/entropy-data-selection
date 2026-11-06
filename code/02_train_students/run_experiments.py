"""Train students on the artifact and score them.

Every training run in the paper is the same experiment with different settings,
so they all live here. A group in `config.json` names a set of students, a set
of arms, and a set of training subsets; the runner takes the cross product and
works through it. Adding a condition means editing JSON, not Python.

    python run_experiments.py --list
    python run_experiments.py --group main_grid
    python run_experiments.py --group sham --dry-run

Each point writes `results/grid/<tag>.json`, holding the accuracy, the training
metadata, and the generated text for every test item. Keeping the generations
is what lets anyone recompute an accuracy, a paired test or a per-subset
spread later without touching a GPU. A point whose result file already exists
is skipped, so an interrupted group picks up where it stopped.

The three arms of the main comparison are deliberately identical except for
what the loss sees:

    text      the same sequences, cross-entropy only          (K = 0)
    artifact  the same sequences plus the gated top-K view    (gate: high)
    low-gate  the same sequences plus the *complement* view   (gate: low)

so a difference between them cannot come from the data, the schedule, the
optimiser or the seed.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdstore import (LogitStore, Report, build_dataloader, censored_kd_loss,
                     gate_mask, load_config, load_pairs, load_written_pairs,
                     required)
from kdstore.models import (add_lora, base_model_of, load_model,
                            load_tokenizer, log, set_seed, vram_needed,
                            wait_for_vram)
from kdstore.paths import prepare, resolve

HERE = Path(__file__).resolve().parent
POSITION_CHUNK = 256


# ------------------------------------------------------------------ planning

def expand(name, group, defaults):
    """Turn one group of the config into a flat list of training points.

    A group is a cross product of students, arms and training subsets. A
    subset entry is either a path, in which case its label is the last
    underscore-separated piece of the filename, or an object carrying its own
    label, which the budget group needs, since four of its subsets end in
    the same seed.
    """
    base = {**defaults, **group.get("defaults", {})}
    points = []
    for student in group["students"]:
        for arm in group["arms"]:
            for entry in group["subsets"]:
                if isinstance(entry, dict):
                    path, label = entry["path"], entry["label"]
                    # any other key on a subset entry overrides the point, so
                    # a subset can carry its own store or its own corpus
                    extra = {k: v for k, v in entry.items()
                             if k not in ("path", "label")}
                else:
                    path, label = entry, Path(entry).stem.split("_")[-1]
                    extra = {}
                point = {**base, **student, **arm, **extra}
                point["subset"] = path
                point["subset_label"] = label
                point["group"] = name
                point["tag"] = "_".join(
                    str(p) for p in (group.get("prefix", name),
                                     student["label"], arm["name"], label))
                points.append(point)
    return points


# ------------------------------------------------------------------ training

def train_one(point, report):
    """Train a single student and return (model, tokenizer, metadata)."""
    device = point.get("device", "cuda")
    set_seed(point.get("seed", 1234))

    k = int(point.get("k", 0))
    store = None
    if k > 0:
        store = LogitStore(resolve(point["store"]))
        if k > store.k_max:
            raise ValueError(f"{point['tag']}: store holds K={store.k_max}")

    if point.get("corpus_file"):
        # Training on text the teacher wrote. Without a subset the whole file
        # is used, which is the case when the file already is the subset. With
        # one, its indices point into this corpus instead of into the public
        # one, that is how a selection rule is applied to rewritten text.
        pairs = load_written_pairs(resolve(point["corpus_file"]))
        if point.get("corpus_subset"):
            indices = json.loads(resolve(point["corpus_subset"]).read_text())["idx"]
        else:
            indices = list(range(len(pairs)))
    else:
        pairs = load_pairs(required(point, "dataset", point["tag"]))["train"]
        indices = json.loads(resolve(point["subset"]).read_text())["idx"]
    tok = load_tokenizer(point["model"], padding_side="right")
    data, loader = build_dataloader(
        pairs, indices, tok, store=store, k=k,
        batch_size=point.get("batch_size", 8),
        gate_source=point.get("gate_source", "entropy"),
        gate_seed=point.get("gate_seed", 1234))

    model = load_model(point["model"], train=True, device=device)
    if point.get("lora"):
        model = add_lora(model, point["lora"])
    body = base_model_of(model)

    accum = point.get("accumulate", 2)
    epochs = point.get("epochs", 2)
    lr = point.get("lr", 2e-4)
    steps = max(1, math.ceil(len(loader) / accum) * epochs)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.06,
        anneal_strategy="cos")

    tau = point.get("tau", float("inf"))
    direction = point.get("gate", "high")
    lam = point.get("lam", 1.0)
    temperature = point.get("temperature", 1.0)
    log(f"{point['tag']}: n={len(data)} K={k} tau={tau} gate={direction} "
        f"lam={lam} T={temperature} steps={steps}")

    gated_total = positions_total = 0
    for epoch in range(epochs):
        # Accumulate totals and divide once. Averaging per-batch averages would
        # over-weight batches that happen to hold few gated positions.
        loss_sum = batches = 0.0
        for step, (ids, att, lab, topk_ids, topk_probs, H) in enumerate(loader):
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            H = H.to(device)
            if topk_ids is not None:
                topk_ids = topk_ids.to(device)
                topk_probs = topk_probs.to(device)

            hidden = body.model(input_ids=ids,
                                attention_mask=att).last_hidden_state
            labelled = lab != -100
            # the prediction for position j is made at j-1
            source = torch.zeros_like(labelled)
            source[:, :-1] = labelled[:, 1:]

            states = hidden[source]
            gold = lab[labelled]
            entropy = H[source]
            ids_k = topk_ids[source] if topk_ids is not None else None
            probs_k = topk_probs[source] if topk_probs is not None else None
            gate = (gate_mask(entropy, tau, direction) if ids_k is not None
                    else torch.zeros_like(entropy, dtype=torch.bool))
            gated_total += int(gate.sum())
            positions_total += int(gate.numel())

            total = torch.zeros((), device=device)
            n = states.shape[0]
            for c in range(0, n, POSITION_CHUNK):
                sl = slice(c, min(c + POSITION_CHUNK, n))

                def one_chunk(h, g, ti, tp, gt):
                    logits = body.lm_head(h).float()
                    summed, kd, _ = censored_kd_loss(
                        logits, g, ti, tp, gt, lam, temperature)
                    return torch.stack([summed, kd.detach()])

                out = checkpoint(
                    one_chunk, states[sl], gold[sl],
                    ids_k[sl] if ids_k is not None else None,
                    probs_k[sl] if probs_k is not None else None,
                    gate[sl], use_reentrant=False)
                total = total + out[0]

            loss = total / max(1, n)
            (loss / accum).backward()
            loss_sum += float(loss)
            batches += 1
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

        log(f"  {point['tag']} epoch {epoch}: loss "
            f"{loss_sum / max(1, batches):.4f} "
            f"rho_gate={gated_total / max(1, positions_total):.3f}")

    model.config.use_cache = True
    info = {"n_train": len(data), "steps": steps, "lora": point.get("lora", 0),
            "rho_gate": gated_total / max(1, positions_total)}
    return model, tok, info


# ---------------------------------------------------------------- evaluation

def evaluate(model, tokenizer, point):
    """Score one trained student on the test split."""
    from kdstore import generate_and_score, load_pairs, load_written_pairs

    if point.get("lora") and hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    # A corpus of one's own is evaluated from a file, the same written-pairs
    # format the training side already accepts, so that nothing has to be
    # registered in kdstore.data to run the pipeline on new data. Scoring is
    # still the exact-match rule in kdstore.evaluation.
    if point.get("eval_file"):
        pairs = load_written_pairs(resolve(point["eval_file"]))
    else:
        pairs = load_pairs(required(point, "dataset", point["tag"]))["test"]
    if point.get("eval_limit"):
        pairs = pairs[:point["eval_limit"]]
    return generate_and_score(
        model, tokenizer, pairs, device=point.get("device", "cuda"),
        batch_size=point.get("eval_batch_size", 24),
        max_new_tokens=point.get("max_new_tokens", 320),
        label=point["tag"], log=log)


# ---------------------------------------------------------------------- main

def run_point(point, results_dir, report, dry_run=False):
    destination = prepare(results_dir / f"{point['tag']}.json")
    if destination.exists():
        existing = json.loads(destination.read_text())
        return existing["acc"] * 100, "cached"
    if dry_run:
        return None, "planned"

    if point.get("device", "cuda") != "cpu":
        wait_for_vram(vram_needed(point["model"]), log=log)
    model, tok, info = train_one(point, report)
    scored = evaluate(model, tok, point)
    record = {
        "tag": point["tag"], "group": point["group"],
        "settings": {key: point[key] for key in
                     ("model", "k", "tau", "gate", "gate_source", "lam",
                      "temperature", "lora", "lr", "epochs", "dataset",
                      "subset", "store", "corpus_file", "corpus_subset",
                      "eval_file")
                     if key in point},
        "train_info": info, **scored,
    }
    destination.write_text(json.dumps(record, ensure_ascii=False))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scored["acc"] * 100, "trained"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--group", action="append",
                    help="group to run; repeatable, default is every group")
    ap.add_argument("--list", action="store_true",
                    help="show the groups and how many points each holds")
    ap.add_argument("--dry-run", action="store_true",
                    help="expand the plan and print it without training")
    ap.add_argument("--shard", metavar="i/n",
                    help="run only every n-th point, starting at i (1-based). "
                         "Several shards can run side by side on one card; "
                         "each skips points another shard has finished, so "
                         "they do not collide.")
    args = ap.parse_args()

    config = load_config(args.config)
    defaults = config.get("defaults", {})
    chosen = args.group or list(config["groups"])

    if args.list:
        report = Report("Training groups")
        rows = []
        for name, group in config["groups"].items():
            points = expand(name, group, defaults)
            rows.append([name, len(points), group.get("description", "")])
        report.table("groups", ["group", "points", "what it measures"], rows)
        return

    results_dir = resolve(config.get("results_dir", "results/grid"))
    report = Report(f"Training: {', '.join(chosen)}")

    for name in chosen:
        if name not in config["groups"]:
            raise SystemExit(f"no group named {name!r}")
        group = config["groups"][name]
        points = expand(name, group, defaults)
        if args.shard:
            i, n = (int(x) for x in args.shard.split("/"))
            points = points[i - 1::n]
        report.note(f"\n  --- {name}: {group.get('description', '')} ---")

        rows = []
        for point in points:
            accuracy, state = run_point(point, results_dir, report,
                                        args.dry_run)
            rows.append([point["tag"], point.get("k", 0), point.get("gate"),
                         point["subset_label"], accuracy, state])
        report.table(f"group_{name}",
                     ["tag", "K", "gate", "subset", "accuracy", "status"],
                     rows, title=f"{name}: {len(points)} points")

    out = resolve(config.get("report_dir", "results"))
    stem = "train_students"
    if args.group:
        stem += "_" + "_".join(args.group)
    if args.shard:
        stem += "_shard" + args.shard.replace("/", "of")
    report.save(out / f"{stem}.json", out / f"{stem}.txt")


if __name__ == "__main__":
    main()
