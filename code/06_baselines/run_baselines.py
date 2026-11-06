"""The two baselines the artifact has to be measured against.

Both are ways of moving a teacher's knowledge to a student that do not involve
publishing a logit store, and both have to be beaten, or at least matched, for
the artifact to be large enough to matter.

    sequence_kd     the teacher writes out solutions to the same prompts and
                    the student trains on that text with plain cross-entropy
    full_logit_kd   the teacher runs alongside the student and supervises the
                    whole 151,936-way distribution at the gated positions

    python run_baselines.py --baseline sequence_kd
    python run_baselines.py --baseline full_logit_kd --shard 1/2

Sequence-level distillation \\cite{kim2016seqkd} is the sharper comparison of
the two. Its output is also a publishable file, a few hundred kilobytes of
text, and unlike a logit store it carries across tokenizers, so it is
portable in a stronger sense than the artifact is. Anything the artifact
offers has to be set against that.

Full-logit distillation is the ceiling: it is what the store is an
approximation of. It needs the teacher resident in memory for every step,
which is the cost the artifact exists to avoid, so it is not an alternative to
the artifact. It bounds what the approximation gives up.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdstore import (LogitStore, Report, build_dataloader, gate_mask,
                     generate_and_score, load_config, load_pairs, required,
                     score_generations)
from kdstore.models import (MAX_LENGTH, PROMPT, add_lora, base_model_of,
                            load_model, load_tokenizer, log, set_seed,
                            vram_needed, wait_for_vram)
from kdstore.paths import prepare, resolve

HERE = Path(__file__).resolve().parent
POSITION_CHUNK = 256


# ------------------------------------------------------- teacher generations

@torch.no_grad()
def generate_solutions(spec, subset_path, report):
    """Have the teacher write a solution for every prompt in a subset.

    Cached: the file is written once and reused, because this is the expensive
    half of the baseline and it does not depend on the student.
    """
    destination = prepare(resolve(spec["generations_dir"])
                          / f"{Path(subset_path).stem}.json")
    if destination.exists():
        payload = json.loads(destination.read_text())
        log(f"  reusing {destination.name}: {len(payload['pairs'])} solutions")
        return payload["pairs"], payload["accuracy"]

    device = spec.get("device", "cuda")
    tokenizer = load_tokenizer(spec["tokenizer"], padding_side="left")
    teacher = load_model(resolve(spec["teacher"]), train=False, device=device)

    corpus = load_pairs(required(spec, "dataset"))["train"]
    indices = json.loads(resolve(subset_path).read_text())["idx"]
    prompts = [corpus[i][0] for i in indices]
    references = [corpus[i][1] for i in indices]

    batch = spec.get("generate_batch_size", 16)
    written = []
    for start in range(0, len(prompts), batch):
        chunk = prompts[start:start + batch]
        enc = tokenizer([PROMPT.format(q=q) for q in chunk],
                        return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_LENGTH,
                        add_special_tokens=False).to(device)
        out = teacher.generate(**enc,
                               max_new_tokens=spec.get("max_new_tokens", 320),
                               do_sample=False,
                               pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id)
        written += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        if (start // batch) % 5 == 0:
            log(f"  teacher wrote {len(written)}/{len(prompts)}")

    # How often the teacher's own solution reaches the right answer. The
    # student is trained on all of them either way, filtering would be a
    # different method, but a baseline whose targets are wrong half the time
    # is not the same baseline.
    scored = score_generations(written, references)
    pairs = [[q, text] for q, text in zip(prompts, written)]

    del teacher
    torch.cuda.empty_cache()
    destination.write_text(json.dumps(
        {"subset": str(spec["subset"]), "teacher": str(spec["teacher"]),
         "accuracy": scored["acc"], "pairs": pairs}, ensure_ascii=False))
    log(f"  wrote {destination.name}; teacher solutions are correct "
        f"{scored['acc'] * 100:.1f}% of the time")
    return pairs, scored["acc"]


# ------------------------------------------------------------- sequence-level

def run_sequence_kd(spec, point, report):
    """Plain cross-entropy on the teacher's own solutions."""
    device = spec.get("device", "cuda")
    set_seed(point.get("seed", 1234))

    generated, teacher_accuracy = generate_solutions(spec, point["subset"],
                                                     report)
    tokenizer = load_tokenizer(point["model"], padding_side="right")
    # The dataset indexes into `pairs`, so hand it the generated corpus and
    # ask for all of it.
    data, loader = build_dataloader(
        [tuple(x) for x in generated], range(len(generated)), tokenizer,
        store=None, k=0, batch_size=point.get("batch_size", 8))

    model = load_model(point["model"], train=True, device=device)
    if point.get("lora"):
        model = add_lora(model, point["lora"])
    body = base_model_of(model)

    accum = point.get("accumulate", 2)
    epochs = point.get("epochs", 2)
    lr = point.get("lr", 2e-4)
    steps = max(1, math.ceil(len(loader) / accum) * epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.06,
        anneal_strategy="cos")

    log(f"{point['tag']}: n={len(data)} sequence-level, steps={steps}")
    for epoch in range(epochs):
        total, batches = 0.0, 0
        for step, (ids, att, lab, _, _, _) in enumerate(loader):
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / accum).backward()
            total += float(loss)
            batches += 1
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        log(f"  {point['tag']} epoch {epoch}: loss {total / max(1, batches):.4f}")

    model.config.use_cache = True
    return model, tokenizer, {"n_train": len(data), "steps": steps,
                              "teacher_solution_accuracy": teacher_accuracy}


# ------------------------------------------------------------- full-logit KD

def run_full_logit_kd(spec, point, report):
    """The teacher supervises the whole distribution, online.

    Same gate, same positions, same schedule as the artifact arm; the only
    change is that the divergence is taken over all 151,936 tokens instead of
    over the K the store would hold. That makes it the ceiling the artifact is
    approximating, and it is the one arm here that cannot be shipped as a
    file.
    """
    device = spec.get("device", "cuda")
    set_seed(point.get("seed", 1234))

    store = LogitStore(resolve(point["store"]))
    corpus = load_pairs(required(point, "dataset", point["tag"]))["train"]
    indices = json.loads(resolve(point["subset"]).read_text())["idx"]
    tokenizer = load_tokenizer(point["model"], padding_side="right")
    # k=1 only so the dataset carries the entropy channel the gate reads; the
    # stored probabilities themselves are not used here.
    data, loader = build_dataloader(corpus, indices, tokenizer, store=store,
                                    k=1,
                                    batch_size=point.get("batch_size", 8))

    teacher = load_model(resolve(spec["teacher"]), train=False, device=device)
    teacher_body = base_model_of(teacher)
    model = load_model(point["model"], train=True, device=device)
    if point.get("lora"):
        model = add_lora(model, point["lora"])
    body = base_model_of(model)

    accum = point.get("accumulate", 2)
    epochs = point.get("epochs", 2)
    lr = point.get("lr", 2e-4)
    tau = point.get("tau", 0.0335)
    direction = point.get("gate", "high")
    lam = point.get("lam", 1.0)
    steps = max(1, math.ceil(len(loader) / accum) * epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.06,
        anneal_strategy="cos")

    log(f"{point['tag']}: n={len(data)} full-logit, tau={tau} steps={steps}")
    gated_total = positions_total = 0
    for epoch in range(epochs):
        running, batches = 0.0, 0
        for step, (ids, att, lab, _, _, H) in enumerate(loader):
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            H = H.to(device)

            labelled = lab != -100
            source = torch.zeros_like(labelled)
            source[:, :-1] = labelled[:, 1:]
            gold = lab[labelled]
            gate = gate_mask(H[source], tau, direction)
            gated_total += int(gate.sum())
            positions_total += int(gate.numel())

            with torch.no_grad():
                teacher_hidden = teacher_body.model(
                    input_ids=ids, attention_mask=att).last_hidden_state[source]
            hidden = body.model(input_ids=ids,
                                attention_mask=att).last_hidden_state[source]

            total = torch.zeros((), device=device)
            n = hidden.shape[0]
            for c in range(0, n, POSITION_CHUNK):
                sl = slice(c, min(c + POSITION_CHUNK, n))

                def one_chunk(student_h, teacher_h, g, gt):
                    student = body.lm_head(student_h).float()
                    with torch.no_grad():
                        target = torch.log_softmax(
                            teacher_body.lm_head(teacher_h).float(), -1)
                    ce = F.cross_entropy(student, g, reduction="none")
                    kd = torch.zeros_like(ce)
                    if gt.any():
                        kd[gt] = F.kl_div(
                            torch.log_softmax(student[gt], -1), target[gt],
                            reduction="none", log_target=True).sum(-1)
                    return torch.where(gt, lam * kd + (1 - lam) * ce, ce).sum()

                total = total + checkpoint(
                    one_chunk, hidden[sl], teacher_hidden[sl], gold[sl],
                    gate[sl], use_reentrant=False)

            loss = total / max(1, n)
            (loss / accum).backward()
            running += float(loss)
            batches += 1
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        log(f"  {point['tag']} epoch {epoch}: loss "
            f"{running / max(1, batches):.4f} "
            f"rho_gate={gated_total / max(1, positions_total):.3f}")

    del teacher
    torch.cuda.empty_cache()
    model.config.use_cache = True
    return model, tokenizer, {"n_train": len(data), "steps": steps,
                              "rho_gate": gated_total / max(1, positions_total)}


# ---------------------------------------------------------------------- main

BASELINES = {"sequence_kd": run_sequence_kd, "full_logit_kd": run_full_logit_kd}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--baseline", action="append",
                    help="repeatable; default is every baseline in the config")
    ap.add_argument("--shard", metavar="i/n")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    chosen = args.baseline or list(config["baselines"])
    results_dir = resolve(config.get("results_dir", "results/grid"))
    report = Report(f"Baselines: {', '.join(chosen)}")

    for name in chosen:
        if name not in config["baselines"]:
            raise SystemExit(f"no baseline named {name!r} in {args.config}")
        spec = {**config.get("defaults", {}), **config["baselines"][name]}
        # `kind` lets one config hold the same baseline on several cells
        kind = spec.get("kind", name)
        points = []
        for student in spec["students"]:
            for entry in spec["subsets"]:
                if isinstance(entry, dict):
                    subset, label = entry["path"], entry["label"]
                    extra = {k: v for k, v in entry.items()
                             if k not in ("path", "label")}
                else:
                    subset = entry
                    label = Path(subset).stem.split("_")[-1]
                    extra = {}
                points.append({**spec, **student, **extra, "subset": subset,
                               "tag": f"{spec.get('prefix', name)}_"
                                      f"{student['label']}_{label}",
                               "subset_label": label})
        if args.shard:
            i, n = (int(x) for x in args.shard.split("/"))
            points = points[i - 1::n]

        report.note(f"\n  --- {name}: {spec.get('description', '')} ---")
        rows = []
        for point in points:
            destination = prepare(results_dir / f"{point['tag']}.json")
            if destination.exists():
                rows.append([point["tag"], point["subset_label"],
                             json.loads(destination.read_text())["acc"] * 100,
                             "cached"])
                continue
            if args.dry_run:
                rows.append([point["tag"], point["subset_label"], None,
                             "planned"])
                continue

            if point.get("device", "cuda") != "cpu":
                # full-logit distillation holds the teacher as well
                extra = 9000 if kind == "full_logit_kd" else 0
                wait_for_vram(vram_needed(point["model"], extra=extra),
                              log=log)
            model, tokenizer, info = BASELINES[kind](spec, point, report)
            if point.get("lora") and hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
            pairs = load_pairs(required(point, "dataset", point["tag"]))["test"]
            scored = generate_and_score(
                model, tokenizer, pairs, device=point.get("device", "cuda"),
                batch_size=point.get("eval_batch_size", 24),
                label=point["tag"], log=log)
            destination.write_text(json.dumps(
                {"tag": point["tag"], "group": name, "train_info": info,
                 "settings": {k: point[k] for k in
                              ("model", "lora", "lr", "epochs", "dataset",
                               "subset", "tau", "gate") if k in point},
                 **scored}, ensure_ascii=False))
            rows.append([point["tag"], point["subset_label"],
                         scored["acc"] * 100, "trained"])
            del model
            torch.cuda.empty_cache()

        report.table(name, ["tag", "subset", "accuracy", "status"], rows,
                     title=spec.get("description", name))

    out = resolve(config.get("report_dir", "results"))
    stem = "baselines" + ("_" + "_".join(chosen) if args.baseline else "")
    stem += "_shard" + args.shard.replace("/", "of") if args.shard else ""
    report.save(out / f"{stem}.json", out / f"{stem}.txt")


if __name__ == "__main__":
    main()
