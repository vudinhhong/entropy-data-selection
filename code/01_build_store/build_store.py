"""Build the materials: corpus split, teacher, logit store, sham stores.

This is the only group that has to run before any other. It takes a corpus,
cuts it in half, fine-tunes a teacher on one half, runs that teacher over the
other half once, and writes down what it predicted. Everything the paper
measures afterwards reads the file this produces.

    python build_store.py --stage all
    python build_store.py --stage store --only gsm8k

Stages
    splits   halve the corpus, then draw the training subsets
    teacher  fine-tune the teacher on the half it is allowed to see
    store    one forward pass; writes the store and its metadata
    sham     derive the two control stores from a real one
    verify   integrity checks that should pass before anyone trains

The teacher half and the student half never overlap. That is the point: the
teacher's predictions on the student half are predictions about text it was not
trained on, which is why its entropy there is 0.1308 nats against 0.0573 on its
own half.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdstore import (LogitStore, Report, load_config, load_pairs,
                     load_written_pairs)
from kdstore.models import (MAX_LENGTH, encode, add_lora, load_model,
                            load_tokenizer, log, set_seed, vram_needed,
                            wait_for_vram)
from kdstore.paths import prepare, resolve

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------- splits

def make_splits(spec, report):
    """Halve the corpus and draw the training subsets.

    The split is a permutation under a fixed seed, so anyone can regenerate it.
    Subsets are drawn from the student half only, and they overlap, three
    draws of 10% from the same pool share some rows, which is exactly why the
    paired test treats the subset as the unit and not as an independent sample.
    """
    pairs = load_pairs(spec["dataset"])["train"]
    n = len(pairs)
    rng = np.random.default_rng(spec.get("split_seed", 1234))
    order = rng.permutation(n)
    # An even split is right when the corpus is small enough that the teacher
    # needs half of it. On a corpus of hundreds of thousands it only starves
    # the pool the student selects from, so the share is a config knob.
    half = int(round(n * spec.get("teacher_share", 0.5)))
    teacher_half = sorted(int(i) for i in order[:half])
    student_half = sorted(int(i) for i in order[half:])

    out = resolve(spec["artifacts"])
    written = []
    for name, idx in (("split_A", teacher_half), ("split_B", student_half)):
        path = prepare(out / f"{name}{spec.get('suffix', '')}.json")
        if path.exists() and not spec.get("redraw_splits"):
            # Never silently replace a split that stores are already indexed
            # against: every offset in every store, and every subset drawn so
            # far, refers to this exact partition. Reuse it, and draw any new
            # subsets from the pool it defines instead of from a fresh one.
            existing = json.loads(path.read_text())["idx"]
            if name == "split_B":
                student_half = existing
            written.append((name, len(existing), path.name + " (kept)"))
            continue
        path.write_text(json.dumps(
            {"idx": idx,
             "info": {"dataset": spec["dataset"], "role":
                      "teacher" if name == "split_A" else "student",
                      "n": len(idx), "of": n,
                      "seed": spec.get("split_seed", 1234)}}))
        written.append((name, len(idx), path.name))

    fraction = spec.get("subset_fraction", 0.10)
    size = max(1, round(fraction * len(student_half)))
    for seed in spec.get("subset_seeds", [11, 22, 1234]):
        draw = np.random.default_rng(seed).choice(
            student_half, size=size, replace=False)
        path = prepare(out / f"{spec.get('subset_prefix', 'subset')}_s{seed}.json")
        if path.exists() and not spec.get("redraw_subsets"):
            written.append((f"subset s{seed}",
                            len(json.loads(path.read_text())["idx"]),
                            path.name + " (kept)"))
            continue
        path.write_text(json.dumps(
            {"idx": sorted(int(i) for i in draw),
             "info": {"drawn_from": "split_B", "fraction": fraction,
                      "seed": seed, "n": size}}))
        written.append((f"subset s{seed}", size, path.name))

    report.table(f"splits_{spec['name']}",
                 ["file", "sequences", "written as"],
                 [[a, b, c] for a, b, c in written],
                 title=f"{spec['name']}: corpus split")
    return teacher_half, student_half


# ------------------------------------------------------------------- teacher

def train_teacher(spec, report):
    """Fine-tune the teacher on split A, supervised on answer tokens only."""
    from torch.utils.data import DataLoader, Dataset

    import math

    cfg = spec["teacher"]
    device = spec.get("device", "cuda")
    set_seed(spec.get("seed", 1234))

    pairs = load_pairs(spec["dataset"])["train"]
    split = json.loads(
        (resolve(spec["artifacts"]) / f"split_A{spec.get('suffix','')}.json")
        .read_text())["idx"]
    tok = load_tokenizer(cfg["model"])

    class TextOnly(Dataset):
        def __init__(self):
            self.items = [encode(tok, *pairs[i]) for i in split]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        ids_t = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        lab = torch.full((len(batch), width), -100, dtype=torch.long)
        for r, (ids, prompt_len) in enumerate(batch):
            ids_t[r, :len(ids)] = torch.tensor(ids)
            att[r, :len(ids)] = 1
            lab[r, prompt_len:len(ids)] = torch.tensor(ids[prompt_len:])
        return ids_t, att, lab

    data = TextOnly()
    loader = DataLoader(data, batch_size=cfg.get("batch_size", 8), shuffle=True,
                        collate_fn=collate)
    model = load_model(cfg["model"], train=True, device=device)
    if cfg.get("lora"):
        model = add_lora(model, cfg["lora"])
        model.print_trainable_parameters()

    accum = cfg.get("accumulate", 2)
    epochs = cfg.get("epochs", 3)
    steps = max(1, math.ceil(len(loader) / accum) * epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-5),
                            weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.get("lr", 1e-5), total_steps=steps,
        pct_start=cfg.get("warmup", 0.03), anneal_strategy="cos")

    log(f"teacher {spec['name']}: {len(data)} sequences, {steps} updates")
    for epoch in range(epochs):
        running, seen = 0.0, 0
        for i, (ids, att, lab) in enumerate(loader):
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / accum).backward()
            running += float(loss)
            seen += 1
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        log(f"  epoch {epoch}: loss {running / max(1, seen):.4f}")

    if cfg.get("lora"):
        model = model.merge_and_unload()
    model.config.use_cache = True
    out = prepare(resolve(cfg["output"]) / "config.json").parent
    model.save_pretrained(out)
    tok.save_pretrained(out)
    report.value(f"teacher_{spec['name']}", str(out), "teacher saved to")
    return out


# --------------------------------------------------------------------- store

@torch.no_grad()
def build_store(spec, report):
    """One teacher forward pass over the whole corpus; write top-K and entropy.

    Written incrementally to a `.partial` file and resumable. A pass over a 4B
    teacher is twenty-odd minutes of GPU and everything downstream depends on
    it, so losing it halfway is expensive.

    Probabilities are stored in log space. Same sixteen bits, but float16
    bottoms out at 6e-8, so storing them raw zeroes the tail wherever the
    teacher is confident, which is most positions. In log space a probability
    of 1e-13 is -29.9, comfortably inside range, and every ratio survives.
    """
    cfg = spec["store"]
    device = spec.get("device", "cuda")
    k_max = cfg.get("k_max", 50)
    chunk = cfg.get("position_chunk", 512)
    flush_every = cfg.get("flush_batches", 40)
    batch_size = cfg.get("batch_size", 8)
    set_seed(spec.get("seed", 1234))

    if device != "cpu":
        # Size the wait from the teacher actually configured. A path like
        # runs/teacher_gsm8k carries no size and keeps the 4B default; a model
        # id does, so a smaller teacher is not made to wait for room it will
        # never use.
        wait_for_vram(vram_needed(cfg.get("teacher", ""), default=13000),
                      log=log)
    # Resolve before testing. A config path is relative to the release root,
    # not to the working directory, so the raw string must not be tested.
    teacher_dir = resolve(cfg["teacher"])
    # A teacher we fine-tuned is a directory under the release root. A teacher
    # taken off the shelf is a hub id, which is not a path and must not be
    # resolved into one: resolving "Qwen/Qwen2.5-0.5B" produces a directory
    # that does not exist, and the loader then reports it as a malformed repo
    # id. Prefer the directory when it is there, and otherwise pass the string
    # through untouched.
    teacher_source = str(teacher_dir) if teacher_dir.exists() else cfg["teacher"]
    tokenizer_id = cfg.get("tokenizer") or teacher_source \
        or spec.get("teacher", {}).get("model")
    if tokenizer_id is None:
        raise SystemExit(
            f"{spec['name']}: cannot find a tokenizer. Give the store spec a "
            f"'tokenizer', or point 'teacher' at a directory that exists.")
    tok = load_tokenizer(tokenizer_id)
    model = load_model(teacher_source, train=False, device=device)
    vocab = model.config.vocab_size

    # Normally the corpus is the public training split. It can also be text
    # the teacher wrote itself, which is how we measure what a teacher's
    # distribution looks like on its own greedy output.
    if cfg.get("corpus_file"):
        pairs = load_written_pairs(resolve(cfg["corpus_file"]))
        log(f"  corpus: {len(pairs)} sequences from "
            f"{Path(cfg['corpus_file']).name}")
    else:
        pairs = load_pairs(spec["dataset"])["train"]
    limit = cfg.get("limit")
    n = min(limit, len(pairs)) if limit else len(pairs)
    encoded = [encode(tok, *pairs[i]) for i in range(n)]

    out_prefix = resolve(cfg["output"])
    prepare(out_prefix)
    partial = Path(str(out_prefix) + ".partial.npz")

    done = {}
    if partial.exists():
        with np.load(partial) as z:
            offset = 0
            for key, length in zip(z["done"].tolist(), z["lens"].tolist()):
                done[key] = tuple(z[name][offset:offset + length] for name in
                                  ("idx", "logp", "H", "gold_p", "gold_rank"))
                offset += length
            for key in z["empty"].tolist():
                done[key] = None
        log(f"resuming: {len(done)}/{n} sequences already computed")

    def flush():
        if not done:
            return
        keys = sorted(done)
        filled = [k for k in keys if done[k] is not None]
        empty = np.array([k for k in keys if done[k] is None], dtype=np.int64)

        def stack(column, dtype, width=None):
            if not filled:
                shape = (0, width) if width else (0,)
                return np.zeros(shape, dtype)
            return np.concatenate([done[k][column] for k in filled])

        np.savez(partial,
                 done=np.array(filled, dtype=np.int64),
                 lens=np.array([len(done[k][2]) for k in filled], dtype=np.int64),
                 empty=empty,
                 idx=stack(0, np.int32, k_max), logp=stack(1, np.float16, k_max),
                 H=stack(2, np.float16), gold_p=stack(3, np.float16),
                 gold_rank=stack(4, np.int32))

    # Batch sequences of similar length together: padding is wasted compute and
    # the corpus has a long tail of long answers.
    order = sorted(range(n), key=lambda i: len(encoded[i][0]))

    for start in range(0, n, batch_size):
        todo = [i for i in order[start:start + batch_size] if i not in done]
        if not todo:
            continue
        width = max(len(encoded[i][0]) for i in todo)
        ids = torch.full((len(todo), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(todo), width), dtype=torch.long)
        for r, i in enumerate(todo):
            seq = encoded[i][0]
            ids[r, :len(seq)] = torch.tensor(seq)
            att[r, :len(seq)] = 1
        hidden = model.model(input_ids=ids.to(device),
                             attention_mask=att.to(device)).last_hidden_state

        for r, i in enumerate(todo):
            seq, prompt_len = encoded[i]
            # answer token j is predicted at index j-1
            positions = torch.arange(prompt_len - 1, len(seq) - 1, device=device)
            if len(positions) == 0:
                done[i] = None
                continue
            gold = torch.tensor(seq[prompt_len:len(seq)], device=device)
            parts = [[] for _ in range(5)]
            for c in range(0, len(positions), chunk):
                pos = positions[c:c + chunk]
                g = gold[c:c + chunk]
                logits = model.lm_head(hidden[r, pos]).float()
                logp = torch.log_softmax(logits, -1)
                prob = logp.exp()
                top_prob, top_id = prob.topk(k_max, -1)
                parts[0].append(top_id.to(torch.int32))
                parts[1].append(logp.gather(-1, top_id).to(torch.float16))
                parts[2].append((-(prob * logp).sum(-1)).to(torch.float16))
                parts[3].append(prob.gather(-1, g[:, None]).squeeze(-1)
                                .to(torch.float16))
                parts[4].append((logits > logits.gather(-1, g[:, None]))
                                .sum(-1).to(torch.int32))
                del logits, logp, prob
            done[i] = tuple(torch.cat(p).cpu().numpy() for p in parts)
        del hidden

        batch_no = start // batch_size
        if batch_no % 25 == 0:
            log(f"  forward {start}/{n}")
        if batch_no and batch_no % flush_every == 0:
            flush()

    # Reassemble in corpus order so the file is stable across runs.
    columns = [[] for _ in range(5)]
    offsets = [0]
    meta = []
    for i in range(n):
        row = done[i]
        if row is None:
            offsets.append(offsets[-1])
            meta.append({"i": i, "npos": 0})
            continue
        for c in range(5):
            columns[c].append(row[c])
        offsets.append(offsets[-1] + len(row[2]))
        mass = np.exp(row[1].astype(np.float32)).sum(-1)
        meta.append({
            "i": i, "npos": int(len(row[2])),
            "mean_H": float(row[2].astype(np.float32).mean()),
            "mean_nll": float(-np.log(np.clip(row[3].astype(np.float32),
                                              1e-9, None)).mean()),
            "topk_mass": float(mass.mean()),
            "gold_in_topk": float((row[4] < k_max).mean()),
            "steps": pairs[i][1].count("\n"),
        })

    np.savez(str(out_prefix) + ".npz",
             idx=np.concatenate(columns[0]), logp=np.concatenate(columns[1]),
             H=np.concatenate(columns[2]), gold_p=np.concatenate(columns[3]),
             gold_rank=np.concatenate(columns[4]),
             offsets=np.array(offsets, dtype=np.int64))
    Path(str(out_prefix) + ".meta.json").write_text(json.dumps(
        {"n": n, "k_max": k_max, "vocab_size": vocab,
         "teacher": str(resolve(cfg["teacher"])), "encoding": "logp",
         "total_positions": int(offsets[-1]), "meta": meta}))
    if partial.exists():
        partial.unlink()

    report.table(f"store_{spec['name']}", ["property", "value"],
                 [["sequences", n], ["positions", int(offsets[-1])],
                  ["K stored", k_max], ["vocabulary", vocab],
                  ["file", Path(str(out_prefix) + ".npz").name]],
                 title=f"{spec['name']}: logit store")
    return Path(str(out_prefix) + ".npz")


# ---------------------------------------------------------------------- sham

def build_sham(spec, report):
    """The two control stores.

    flat     every top-K probability set to m/K. Keeps the mass, keeps the
             identities, destroys the shape.
    shuffled identities permuted within each position. Keeps the mass, keeps
             the shape, destroys which probability belongs to which token.

    Entropy and offsets are copied unchanged, so the gate selects exactly the
    same positions as it does on the real store. Anything the sham arms lose is
    therefore lost by the content of the distribution, not by the gate.
    """
    cfg = spec["sham"]
    source = resolve(cfg["source"])
    with np.load(source) as z:
        idx, logp = z["idx"], z["logp"]
        carry = {k: z[k] for k in ("H", "gold_p", "gold_rank", "offsets")}

    k = idx.shape[1]
    prob = np.exp(logp.astype(np.float64))
    mass = prob.sum(1, keepdims=True).clip(1e-12, 1.0)

    flat = np.repeat(mass / k, k, axis=1)
    flat_path = prepare(resolve(cfg["flat_output"]))
    np.savez_compressed(flat_path, idx=idx,
                        logp=np.log(flat).astype(logp.dtype), **carry)

    rng = np.random.default_rng(cfg.get("seed", 1234))
    shuffled = idx.copy()
    for row in range(shuffled.shape[0]):
        rng.shuffle(shuffled[row])
    shuffled_path = prepare(resolve(cfg["shuffled_output"]))
    np.savez_compressed(shuffled_path, idx=shuffled, logp=logp, **carry)

    report.table(f"sham_{spec['name']}", ["store", "check", "value"],
                 [["flat", "max mass drift",
                   float(np.abs(flat.sum(1) - mass[:, 0]).max())],
                  ["shuffled", "rows left in original order",
                   float((shuffled == idx).all(1).mean())],
                  ["shuffled", "identity multiset preserved",
                   bool((np.sort(shuffled, 1) == np.sort(idx, 1)).all())]],
                 title=f"{spec['name']}: sham stores")
    return flat_path, shuffled_path


# ------------------------------------------------------------------- smooth

def _probability_for_entropy(target, vocab, lo=None, hi=1.0, iterations=60):
    """Find p such that putting p on one token and spreading the rest evenly
    over the other V-1 gives entropy `target` (nats).

    H(p) = -p ln p - (1-p) ln(1-p) + (1-p) ln(V-1), which decreases
    monotonically from ln V at p = 1/V to 0 at p = 1, so a bisection is exact
    to machine precision in sixty steps.
    """
    lo = lo if lo is not None else 1.0 / vocab

    def entropy(p):
        if p >= 1.0:
            return 0.0
        rest = 1.0 - p
        out = rest * math.log(vocab - 1) - rest * math.log(rest)
        if p > 0:
            out -= p * math.log(p)
        return out

    if target >= entropy(lo):
        return lo
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if entropy(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_smooth(spec, report):
    """A label-smoothing store matched to the teacher's entropy, position by
    position.

    The two sham stores answer "does the shape of the teacher's distribution
    matter, holding its support and its mass fixed". They do not answer the
    separate question of whether any sufficiently soft target would do, because
    both of them move probability onto tokens the teacher thinks are wrong.
    This one does: the gold token keeps the largest share, the rest is spread
    uniformly over the vocabulary, and the share is chosen so that the
    resulting entropy equals the teacher's at that position. It is a softened
    one-hot label that is exactly as uncertain as the teacher was, and it knows
    nothing else about what the teacher believed.

    Needs a tokenizer to recover the gold identity at each position, but no GPU.
    """
    store = LogitStore(resolve(spec["smooth"]["source"]))
    cfg = spec["smooth"]
    vocab = cfg.get("vocab_size", 151936)
    pairs = load_pairs(spec["dataset"])["train"]
    tok = load_tokenizer(cfg["tokenizer"])

    H = store.entropy()
    idx = store.array("idx").copy()
    logp = np.empty_like(store.array(store.encoding))
    k = store.k_max
    offsets = store.array("offsets")

    # one bisection per distinct entropy value, not per position
    unique, inverse = np.unique(np.round(H, 6), return_inverse=True)
    solved = np.array([_probability_for_entropy(float(h), vocab)
                       for h in unique])
    p_gold = solved[inverse]
    log_rest = np.log((1.0 - p_gold) / (vocab - 1))

    logp[:, 0] = np.log(np.clip(p_gold, 1e-30, 1.0))
    logp[:, 1:] = log_rest[:, None]

    mismatched = 0
    for i in range(store.n_sequences):
        start, stop = int(offsets[i]), int(offsets[i + 1])
        if stop == start:
            continue
        ids, prompt_length = encode(tok, *pairs[i])
        answer = ids[prompt_length:]
        if len(answer) != stop - start:
            mismatched += 1
            continue
        gold = np.asarray(answer, dtype=idx.dtype)
        # gold takes slot 0; whatever used to sit there moves out of the way
        displaced = idx[start:stop, 0].copy()
        idx[start:stop, 0] = gold
        for column in range(1, k):
            clash = idx[start:stop, column] == gold
            idx[start:stop, column] = np.where(clash, displaced,
                                               idx[start:stop, column])

    out = prepare(resolve(cfg["output"]))
    np.savez_compressed(
        out, idx=idx, logp=logp.astype(store.array(store.encoding).dtype),
        H=store.array("H"), gold_p=store.array("gold_p"),
        gold_rank=store.array("gold_rank"), offsets=offsets)

    check = np.exp(logp[:, 0].astype(np.float64))
    recovered = np.array([
        -q * math.log(q) - (1 - q) * math.log((1 - q) / (vocab - 1))
        if 0 < q < 1 else 0.0 for q in check[:2000]])
    report.table(f"smooth_{spec['name']}", ["check", "value"],
                 [["positions", store.n_positions],
                  ["sequences whose tokenization did not match", mismatched],
                  ["mean gold probability", float(check.mean())],
                  ["min gold probability", float(check.min())],
                  ["max entropy error over 2,000 positions",
                   float(np.abs(recovered - H[:2000]).max())],
                  ["file", Path(str(out)).name]],
                 title=f"{spec['name']}: entropy-matched label-smoothing store")


# ----------------------------------------------------------------- selection

def build_selections(spec, report):
    """Write subsets chosen by each layer-2 score, at a matched token budget.

    Layer 2 is the part of the selection algorithm that consults the teacher.
    These subsets compare the layer-2 scores against each other at a matched
    token budget.

    Two cautions are built in. Scores that sum over positions prefer long
    sequences, so every arm is cut to the same number of answer positions
    instead of the same number of sequences; otherwise a win would mean
    "chose more text" as easily as "chose better text". And `length` is
    included as a control that prefers long sequences while ignoring the
    teacher entirely, if it wins as much as the entropy scores, the effect
    is length and not information.

    `random` genuinely varies with the seed, so it gets one subset per seed.
    The other scores are deterministic on a fixed pool: one subset each, to be
    trained under several seeds instead.
    """
    from kdstore import annotate_gate, select_subset, selection_summary

    cfg = spec["selection"]
    store = LogitStore(resolve(cfg["store"]))
    meta = json.loads(resolve(cfg["meta"]).read_text())["meta"]
    pool = (json.loads(resolve(cfg["pool"]).read_text())["idx"]
            if cfg.get("pool") else None)
    annotate_gate(meta, store, cfg["tau"])

    out = resolve(spec["artifacts"])
    budget = cfg.get("token_budget")
    # A token budget caps the arms by answer positions and lets the sequence
    # count float; a fraction caps them by sequences. One or the other, never
    # both, or the tighter cap silently decides the comparison.
    fraction = cfg.get("fraction", 1.0)
    rows = []
    for score in cfg["scores"]:
        seeds = cfg["seeds"] if score == "random" else cfg["seeds"][:1]
        for seed in seeds:
            idx, info = select_subset(meta, fraction, score=score, seed=seed,
                                      pool=pool, token_budget=budget)
            summary = selection_summary(idx, meta)
            name = (f"{cfg['prefix']}_{score}_s{seed}" if score == "random"
                    else f"{cfg['prefix']}_{score}")
            path = prepare(out / f"{name}.json")
            if path.exists() and not cfg.get("overwrite"):
                rows.append([name, "kept", summary["sequences"],
                             summary["positions"], summary["gated_positions"]])
                continue
            path.write_text(json.dumps({"idx": idx, "info": {
                **info, "score": score, "seed": seed, "summary": summary}}))
            rows.append([name, "written", summary["sequences"],
                         summary["positions"], summary["gated_positions"]])

    report.table("selections",
                 ["subset", "status", "sequences", "tokens",
                  "gated positions"], rows,
                 title=f"Layer-2 selection at a {budget}-token budget")


def build_fold_selections(spec, report):
    """Make independent selected subsets, so the criterion has replicates.

    Every score in `build_selections` except `random` is deterministic on a
    fixed pool, so the subset it picks is a single object. Training that one
    subset under several seeds measures training noise; it does not measure
    the criterion, and a paired test over those seeds treats one observation
    as many. The rest of this paper takes the training subset as the unit of
    generalisation, and the selection comparison has to be held to the same
    standard.

    So the pool is partitioned first. Each fold is disjoint from the others,
    each score is applied inside each fold independently, and each fold then
    contributes exactly one paired observation: the same fold selected two
    ways. The cost is budget, a fifth of the pool selected at a tenth is a
    much smaller training set than a tenth of the whole pool, which is why
    the fold count is a config knob instead of a constant.
    """
    from kdstore import (annotate_gate, annotate_scores, select_subset,
                         selection_summary)
    from kdstore.selection import step_ladder

    cfg = spec["fold_selection"]
    store = LogitStore(resolve(cfg["store"]))
    meta = json.loads(resolve(cfg["meta"]).read_text())["meta"]
    pool = json.loads(resolve(cfg["pool"]).read_text())["idx"]

    # The threshold is fixed by coverage, not by a number of nats. A nat
    # means different things to teachers of different competence, and
    # carrying GSM8K's 0.0335 over to MATH let the gate through 56% of
    # positions instead of 30%. Writing "coverage:0.297" in the config asks
    # for the quantile of this corpus's own entropy that passes that share.
    tau = cfg["tau"]
    if isinstance(tau, str) and tau.startswith("coverage:"):
        want = float(tau.split(":", 1)[1])
        offsets, entropy = store.array("offsets"), store.entropy()
        allowed = set(pool)
        keep = np.zeros(len(entropy), dtype=bool)
        for r in meta:
            if r["i"] in allowed and r.get("npos", 0) > 0:
                keep[offsets[r["i"]]:offsets[r["i"] + 1]] = True
        tau = float(np.quantile(entropy[keep], 1.0 - want))
        got = float((entropy[keep] > tau).mean())
        report.note(f"  {spec['name']}: tau = {tau:.4f} nats for "
                    f"{100 * got:.1f}% coverage (asked for {100 * want:.1f}%)")
    annotate_gate(meta, store, tau)

    # A published criterion scored by a separate pass gets loaded here, so
    # that it is spent at the same budget, inside the same folds, as ours.
    scores = list(cfg["scores"])
    for extra in cfg.get("external_scores", []):
        path = resolve(extra["path"])
        if not path.exists():
            # One missing input should cost its own arm, not the whole stage:
            # the other scores are independent of it and are often the ones
            # being waited on.
            report.note(f"  {spec['name']}: {path.name} is absent, dropping "
                        f"the {extra['key']} arm")
            scores = [s for s in scores if s != extra["key"]]
            continue
        kept = annotate_scores(meta, path, extra["key"], extra.get("cap"))
        report.note(f"  {spec['name']}: loaded {extra['key']} for {kept} "
                    f"sequences from {path.name}")

    # Equal quotas only mean equal composition if no stratum is wide enough
    # for the scored arm to take its longest members. On a long-tailed corpus
    # the fixed ladder is too coarse for that, so derive one from the pool.
    allowed = set(pool)
    stratify_by = cfg.get("stratify_by", "steps")
    bins = (step_ladder([r for r in meta if r["i"] in allowed],
                        cfg["min_share"], stratify_by)
            if cfg.get("min_share") else None)
    rng = np.random.default_rng(cfg["fold_seed"])
    order = rng.permutation(np.asarray(pool, dtype=np.int64))
    folds = [sorted(int(i) for i in part)
             for part in np.array_split(order, cfg["folds"])]

    out = resolve(spec["artifacts"])
    rows = []
    for f, fold in enumerate(folds):
        for score in scores:
            # The random control must differ from fold to fold as well as from
            # the scored arms, or the control is one draw wearing five hats.
            seed = int(cfg["fold_seed"]) + 1000 * f
            idx, info = select_subset(meta, cfg["fraction"], score=score,
                                      seed=seed, pool=fold, bins=bins,
                                      stratify_by=stratify_by)
            summary = selection_summary(idx, meta)
            name = f"{cfg['prefix']}{f}_{score}"
            path = prepare(out / f"{name}.json")
            if path.exists() and not cfg.get("overwrite"):
                status = "kept"
            else:
                path.write_text(json.dumps({"idx": idx, "info": {
                    **info, "score": score, "seed": seed, "fold": f,
                    "fold_size": len(fold), "summary": summary}}))
                status = "written"
            rows.append([name, status, len(fold), summary["sequences"],
                         summary["positions"], summary["gated_positions"],
                         round(summary["mean_steps"], 2)])

    report.table("fold selections",
                 ["subset", "status", "pool", "sequences", "tokens",
                  "gated positions", "steps"], rows,
                 title=(f"{cfg['folds']} disjoint folds, each selected at "
                        f"{cfg['fraction']:.0%}"
                        + (f", {len(bins) - 1} strata" if bins else "")))

    # The check the ladder exists for: within one fold, every arm should
    # carry the same difficulty composition, because the quotas are equal.
    # A spread here means the strata are too wide and the comparison is
    # confounded, so it is printed whether or not anyone asks.
    by_fold = {}
    for name, _, _, _, _, _, steps in rows:
        fold = name[len(cfg["prefix"]):].split("_")[0]
        by_fold.setdefault(fold, []).append(steps)
    spreads = [[f, round(max(v) - min(v), 2)] for f, v in sorted(by_fold.items())]
    report.table("fold composition check", ["fold", "mean-step spread"],
                 spreads,
                 title="Difference in mean reasoning steps between the arms "
                       "of one fold; should be near zero")


# -------------------------------------------------------------------- verify

def verify(spec, report):
    """Checks worth running before anyone trains on the file."""
    store = LogitStore(resolve(spec["store"]["output"]).with_suffix(".npz")
                       if not str(spec["store"]["output"]).endswith(".npz")
                       else resolve(spec["store"]["output"]))
    summary = store.describe(tau=spec.get("tau"))

    rows = [[key, value] for key, value in summary.items() if key != "path"]
    mass = store.probabilities.sum(-1)
    rows.append(["top-K mass, min", float(mass.min())])
    rows.append(["top-K mass, fraction above 1.0", float((mass > 1.0001).mean())])
    rows.append(["entropy, any NaN", bool(np.isnan(store.entropy()).any())])
    rows.append(["offsets monotone",
                 bool((np.diff(store.array("offsets")) >= 0).all())])
    report.table(f"verify_{spec['name']}", ["check", "value"], rows,
                 title=f"{spec['name']}: store integrity")


# ---------------------------------------------------------------------- main

STAGES = {"splits": make_splits, "teacher": train_teacher,
          "store": build_store, "sham": build_sham, "smooth": build_smooth,
          "selection": build_selections,
          "fold_selection": build_fold_selections, "verify": verify}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--stage", default="all",
                    choices=["all", *STAGES])
    ap.add_argument("--only", help="run one named experiment from the config")
    args = ap.parse_args()

    config = load_config(args.config)
    experiments = [e for e in config["experiments"]
                   if not args.only or e["name"] == args.only]
    if not experiments:
        raise SystemExit(f"no experiment named {args.only!r} in {args.config}")

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    report = Report(f"Build materials, stages: {', '.join(stages)}")

    for spec in experiments:
        spec = {**config.get("defaults", {}), **spec}
        for name in stages:
            if name in spec.get("skip", []):
                report.note(f"  {spec['name']}: skipping {name}")
                continue
            report.note(f"\n  --- {spec['name']}: {name} ---")
            STAGES[name](spec, report)

    out = resolve(config.get("report_dir", "results"))
    # One file per (stages, experiments) combination. Without this a
    # fold_selection run for one corpus replaced the store report for another.
    stem = "build_store_" + "-".join(stages)
    stem += "_" + "_".join(e["name"] for e in experiments)
    report.save(out / f"{stem}.json", out / f"{stem}.txt")


if __name__ == "__main__":
    main()
