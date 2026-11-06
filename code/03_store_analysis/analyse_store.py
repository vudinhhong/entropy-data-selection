"""Measurements on the artifact itself. No student is trained here.

Three questions, none of which needs a training run to answer:

    bytes       how big is the published file against the corpus it travels
                with, at each width, with and without the gate
    censoring   how much of the teacher's distribution does truncation to K
                actually throw away, measured over the full vocabulary
    tempering   can a consumer holding only the artifact temper it correctly

    python analyse_store.py --analysis bytes
    python analyse_store.py --analysis censoring --analysis tempering

`bytes` runs on a laptop. The other two need the teacher on a GPU, because
both are defined over the full 151,936-way distribution and the artifact does
not contain it, this is intentional of measuring them.
"""
import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kdstore import LogitStore, Report, binary_entropy, load_config
from kdstore.paths import prepare, resolve

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------- bytes

def corpus_bytes(spec, report):
    """Raw and compressed size of the corpus text, for the whole train split.

    Computed from the corpus itself when `datasets` is available, which is the
    reproducible path. Failing that, the config carries the values measured
    when the paper was written, and the report says which was used, a number
    whose provenance is unclear is worse than one that is slightly stale.
    """
    try:
        from kdstore import load_pairs
        pairs = load_pairs(spec["dataset"])["train"]
        text = "\n".join(f"{q}\n{a}" for q, a in pairs).encode("utf-8")
        raw, packed = len(text), len(zlib.compress(text, 6))
        report.note(f"  corpus text measured from the {spec['dataset']} "
                    f"train split ({len(pairs)} sequences)")
        return raw, packed, len(pairs), "measured"
    except Exception as exc:
        fallback = spec.get("corpus_bytes")
        if not fallback:
            raise SystemExit(
                f"cannot load the corpus ({exc}) and the config carries no "
                f"'corpus_bytes' fallback")
        report.note(f"  corpus text taken from config (datasets unavailable: "
                    f"{type(exc).__name__})")
        return (fallback["raw"], fallback["compressed"],
                fallback["sequences"], "from config")


def analyse_bytes(spec, report):
    """Artifact size against the corpus it replaces, at each width.

    Both sides are measured raw and after zlib, so text is never compared
    against an uncompressed tensor or the other way round. The gate mask is
    charged at the binary-entropy bound h(rho), exactly as B(tau, K) charges
    it. That is a bound instead of a real encoder, but at five kilobytes it
    does not move a digit.

    The store holds every position, the gate is applied when it is used --
    so the gate has to be applied here too, or the measurement describes a
    file nobody would publish.
    """
    store = LogitStore(resolve(spec["store"]))
    tau = spec["tau"]
    raw_all, packed_all, n_corpus, provenance = corpus_bytes(spec, report)

    # rho on each subset, to see how much it moves; the table itself describes
    # one concrete artifact, so it must use that subset's own rho. Pairing an
    # averaged rho with one subset's N mixes two different things.
    rho_by_subset = {}
    for entry in spec["subsets"]:
        label = entry["label"]
        indices = json.loads(resolve(entry["path"]).read_text())["idx"]
        rows = store.rows_for(indices)
        kept = store.gate(tau, rows, "high")
        rho_by_subset[label] = (len(indices), rows, rows[kept],
                                float(kept.mean()))

    measured = spec.get("measure_subset", spec["subsets"][0]["label"])
    n_sequences, rows, gated_rows, rho = rho_by_subset[measured]
    mask_bytes = len(rows) * binary_entropy(rho) / 8

    def text_share(n):
        return raw_all * n / n_corpus, packed_all * n / n_corpus

    subset_raw, subset_packed = text_share(n_sequences)
    pool_raw, pool_packed = text_share(spec["student_half"])

    report.table("rho", ["subset", "sequences", "positions", "gated", "rho"],
                 [[label, v[0], len(v[1]), len(v[2]), v[3]]
                  for label, v in rho_by_subset.items()],
                 title=f"Gate coverage at tau = {tau}", places=4)

    report.note(f"\n  rho on subset {measured:<34s} {rho:.4f}")
    report.data["values"]["rho_measured"] = rho
    report.value("mask_bits", binary_entropy(rho), "h(rho), bits per position")
    report.value("mask_kb", mask_bytes / 1e3, "gate mask", unit=" KB")
    report.value("corpus_source", provenance, "corpus byte source")
    report.value("text_pool_mb", pool_packed / 1e6,
                 f"student half ({spec['student_half']} seq), compressed",
                 unit=" MB")

    rows_out = []
    record = {}
    for k in spec["widths"]:
        for label, selection in (("gated", gated_rows), ("ungated", rows)):
            ids = np.ascontiguousarray(store.array("idx")[selection, :k])
            values = np.ascontiguousarray(
                store.array(store.encoding)[selection, :k])
            packed = (len(zlib.compress(ids.tobytes(), 6))
                      + len(zlib.compress(values.tobytes(), 6)))
            mask = mask_bytes if label == "gated" else 0.0
            entry = {
                "raw": ids.nbytes + values.nbytes + mask + subset_raw,
                "compressed": packed + mask + subset_packed,
            }
            entry["vs_corpus"] = entry["compressed"] / pool_packed
            record.setdefault(str(k), {})[label] = entry
        gated, ungated = record[str(k)]["gated"], record[str(k)]["ungated"]
        rows_out.append([k, gated["raw"] / 1e6, gated["compressed"] / 1e6,
                         f"{gated['vs_corpus']:.2f}x",
                         ungated["compressed"] / 1e6,
                         f"{ungated['vs_corpus']:.2f}x"])

    report.table("sizes",
                 ["K", "raw MB", "packed MB", "vs corpus",
                  "ungated MB", "ungated vs corpus"],
                 rows_out,
                 title="Published artifact against the corpus it accompanies")

    # The formula in the paper, alongside the realised file, so the two can be
    # compared. B charges 18 bits per identity; the file spends a 32-bit int.
    from kdstore import storage_cost
    formula = [[k, storage_cost(len(rows), rho, k, spec["vocab_size"]) / 1e6,
                record[str(k)]["gated"]["raw"] / 1e6]
               for k in spec["widths"]]
    report.table("formula", ["K", "B(tau,K) MB", "realised raw MB"], formula,
                 title="Accounting formula against the file on disk")

    report.data["values"]["bytes"] = {
        "tau": tau, "measure_subset": measured, "rho": rho,
        "rho_by_subset": {k: v[3] for k, v in rho_by_subset.items()},
        "mask_bytes": mask_bytes, "corpus_source": provenance,
        "text_pool": [pool_raw, pool_packed],
        "text_subset": [subset_raw, subset_packed], "widths": record,
    }


# -------------------------------------------------------------------- export

def export_artifact(spec, report):
    """Write the artifact the size table describes, as a file someone can use.

    The published store holds every position at the full stored width, because
    the gate is applied when the store is read. That is the right thing to
    archive, but it is not the object the paper measures: the artifact at
    (tau, K) is the selected text, plus a mask saying which positions passed,
    plus K entries at those positions and nothing at the others. This writes
    that object out, one directory per width, so the sizes in the size table
    correspond to files instead of to an accounting.

    Each directory holds

        text.json      the selected sequences, as (question, answer) pairs
        mask.bin       one bit per answer position, packed, row-major
        topk.npz       ids and log-probabilities at the passing positions
        manifest.json  tau, K, counts, vocabulary, and the source store

    Needs no GPU.
    """
    store = LogitStore(resolve(spec["store"]))
    tau = spec["tau"]
    root = resolve(spec["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    indices = json.loads(resolve(spec["subset"]).read_text())["idx"]
    rows = store.rows_for(indices)
    keep = store.gate(tau, rows, "high")
    gated = rows[keep]

    try:
        from kdstore import load_pairs
        corpus = load_pairs(spec["dataset"])["train"]
        text = [[corpus[i][0], corpus[i][1]] for i in indices]
    except Exception as exc:
        report.note(f"  corpus unavailable ({type(exc).__name__}); "
                    f"text.json will hold sequence indices only")
        text = None

    packed = np.packbits(keep.astype(np.uint8))
    written = []
    for k in spec["widths"]:
        out = root / f"tau{tau}_K{k}"
        out.mkdir(parents=True, exist_ok=True)
        ids, _ = store.view(gated, k)
        values = store.array(store.encoding)[gated, :k]
        np.savez_compressed(out / "topk.npz",
                            idx=ids.astype(np.int32), logp=values)
        (out / "mask.bin").write_bytes(packed.tobytes())
        (out / "text.json").write_text(json.dumps(
            text if text is not None else {"sequence_indices": indices},
            ensure_ascii=False))
        (out / "manifest.json").write_text(json.dumps({
            "tau": tau, "K": k, "encoding": store.encoding,
            "value_bits": 16, "vocab_size": spec["vocab_size"],
            "sequences": len(indices),
            "answer_positions": int(len(rows)),
            "positions_passing_gate": int(len(gated)),
            "rho": float(keep.mean()),
            "source_store": Path(store.path).name,
            "mask": "one bit per answer position in corpus order, "
                    "packed big-endian with numpy.packbits",
            "rows": "row j of topk.npz is the j-th position that passes the "
                    "gate, in the same order as the set bits of mask.bin",
        }, indent=1))
        # text.json and the mask are written uncompressed so the directory is
        # readable as it stands; the size table in the paper compares like
        # with like, so report the packed size too.
        raw = sum(f.stat().st_size for f in out.iterdir())
        packed_size = sum(
            len(zlib.compress(f.read_bytes(), 6)) for f in out.iterdir())
        written.append([k, len(gated), raw / 1e6, packed_size / 1e6,
                        f"{packed_size / 1e6 / spec['corpus_mb']:.2f}x",
                        out.name])

    report.table("export", ["K", "stored positions", "directory MB",
                            "packed MB", "packed vs corpus", "directory"],
                 written, title=f"Artifacts written to {root.name}/")


# ----------------------------------------------------------------- censoring

def analyse_censoring(spec, report):
    """How much of the teacher's distribution truncation to K discards.

    D_KL(p_T || p_hat_T), where p_T is the teacher's tempered distribution over
    the whole vocabulary and p_hat_T is what the store supports: the K kept
    entries, with the residual mass spread evenly over everything else. Shown
    against H(p_T), because a divergence is only large or small relative to the
    uncertainty it is approximating.

    Measuring top-K mass instead would measure the wrong thing: the mass is
    0.9999 at K = 50 while the divergence is still non-zero, because what the
    truncation destroys is the shape of the tail, not its weight.
    """
    import torch

    from kdstore import load_pairs
    from kdstore.models import MAX_LENGTH, PROMPT, load_model, load_tokenizer

    device = spec.get("device", "cuda")
    tokenizer = load_tokenizer(spec["tokenizer"])
    model = load_model(resolve(spec["teacher"]), train=False, device=device)
    pairs = load_pairs(spec["dataset"])["train"]
    pool = json.loads(resolve(spec["pool"]).read_text())["idx"]

    widths = spec["widths"]
    temperatures = spec["temperatures"]
    totals = {(k, t): [0.0, 0.0, 0] for k in widths for t in temperatures}
    seen = 0

    with torch.no_grad():
        for i in pool[:spec.get("max_sequences", 48)]:
            if seen >= spec["max_positions"]:
                break
            question, answer = pairs[i]
            enc = tokenizer(PROMPT.format(q=question) + answer,
                            return_tensors="pt", truncation=True,
                            max_length=MAX_LENGTH,
                            add_special_tokens=False).to(device)
            logits = model(**enc).logits[0].float()[:-1]
            take = min(logits.shape[0], spec["max_positions"] - seen)
            logits = logits[:take]
            seen += take

            for temperature in temperatures:
                logp = torch.log_softmax(logits / temperature, dim=-1)
                prob = logp.exp()
                entropy = -(prob * logp).sum(-1)
                vocab = prob.shape[-1]
                for k in widths:
                    top, _ = prob.topk(k, dim=-1)
                    mass = top.sum(-1).clamp(max=1.0)
                    tail = (1.0 - mass).clamp(min=0)
                    # p_hat keeps the top K and spreads `tail` over V-K tokens:
                    #   D_KL = -H_tail(p) - tail * log(tail / (V - K))
                    head_entropy = -(top * torch.log(top.clamp(min=1e-30))).sum(-1)
                    tail_entropy = (entropy - head_entropy).clamp(min=0)
                    kl = (-tail_entropy
                          - tail * torch.log((tail / (vocab - k)).clamp(min=1e-30)))
                    acc = totals[(k, temperature)]
                    acc[0] += float(kl.clamp(min=0).sum())
                    acc[1] += float(entropy.sum())
                    acc[2] += take

    rows = []
    for temperature in temperatures:
        for k in widths:
            kl, entropy, n = totals[(k, temperature)]
            if n:
                rows.append([temperature, k, entropy / n, kl / n,
                             f"{kl / entropy * 100:.1f}%"])
    report.table(spec.get("name", "censoring"),
                 ["T", "K", "H(p_T)", "D_KL(p||p_hat)",
                  "D_KL / H"], rows,
                 title=f"Information truncation discards "
                       f"({seen} positions, V = {model.config.vocab_size})")


# ----------------------------------------------------------------- tempering

def analyse_tempering(spec, report):
    """Whether a consumer holding only the artifact can temper it.

    Tempering properly means p_i -> p_i^(1/T) for every i, then renormalising.
    The artifact has the tail's total mass but not the individual p_i inside
    it, so it has to compute (1-m)^(1/T) and treat the tail as one lump. Since
    x -> x^(1/T) is concave and the tail holds V-K terms, the sum of powers
    exceeds the power of the sum, and the gap widens with T.

    Both quantities are computed here from the teacher's full distribution, so
    the comparison is exact instead of argued.
    """
    import torch

    from kdstore import load_pairs
    from kdstore.models import MAX_LENGTH, PROMPT, TARGET, load_model, load_tokenizer

    device = spec.get("device", "cuda")
    k = spec["k"]
    tokenizer = load_tokenizer(spec["tokenizer"])
    model = load_model(resolve(spec["teacher"]), train=False, device=device)
    pairs = load_pairs(spec["dataset"])["train"]
    pool = json.loads(resolve(spec["pool"]).read_text())["idx"]

    totals = {t: [0.0, 0.0, 0] for t in spec["temperatures"]}
    seen = 0
    with torch.no_grad():
        for i in pool[:spec.get("max_sequences", 24)]:
            if seen >= spec["max_positions"]:
                break
            question, answer = pairs[i]
            prompt = tokenizer(PROMPT.format(q=question),
                               add_special_tokens=False)["input_ids"]
            target = tokenizer(TARGET.format(a=answer),
                               add_special_tokens=False)["input_ids"]
            ids = (prompt + target)[:MAX_LENGTH]
            if len(ids) <= len(prompt) + 1:
                continue
            logits = model(input_ids=torch.tensor([ids], device=device)) \
                .logits[0].float()[len(prompt) - 1:-1]
            take = min(logits.shape[0], spec["max_positions"] - seen)
            logits = logits[:take]
            seen += take

            prob = torch.softmax(logits, -1)
            top, _ = prob.topk(k, -1)
            mass = top.sum(-1)
            for temperature in spec["temperatures"]:
                power = 1.0 / temperature
                head = (top ** power).sum(-1)
                true_tail = (prob ** power).sum(-1) - head
                lump_tail = (1.0 - mass).clamp_min(1e-12) ** power
                totals[temperature][0] += float(
                    (true_tail / (head + true_tail)).sum())
                totals[temperature][1] += float(
                    (lump_tail / (head + lump_tail)).sum())
                totals[temperature][2] += take

    rows = []
    for temperature in spec["temperatures"]:
        true_mass, lump_mass, n = totals[temperature]
        ratio = (true_mass / n) / max(lump_mass / n, 1e-12)
        rows.append([temperature, true_mass / n, lump_mass / n,
                     f"{ratio:.1f}x"])
    report.table(spec.get("name", "tempering"),
                 ["T", "true tail mass", "one-bucket form", "understated by"],
                 rows, title=f"Tempering the stored view (K = {k}, "
                             f"{seen} positions)")


# ---------------------------------------------------------------------- main

ANALYSES = {"bytes": analyse_bytes, "censoring": analyse_censoring,
            "tempering": analyse_tempering, "export": export_artifact}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--analysis", action="append",
                    help="repeatable; default is every analysis in the config")
    args = ap.parse_args()

    config = load_config(args.config)
    chosen = args.analysis or list(config["analyses"])
    report = Report(f"Store analysis: {', '.join(chosen)}")

    for name in chosen:
        if name not in config["analyses"]:
            raise SystemExit(f"no analysis named {name!r} in {args.config}")
        spec = {**config.get("defaults", {}), **config["analyses"][name]}
        # `kind` lets one config hold several analyses of the same type, the
        # censoring measurement, for instance, has to be run once per teacher.
        kind = spec.get("kind", name)
        spec["name"] = name
        report.note(f"\n  --- {name} ({kind}) ---")
        ANALYSES[kind](spec, report)

    out = resolve(config.get("report_dir", "results"))
    stem = "store_analysis" if not args.analysis else \
        "store_analysis_" + "_".join(chosen)
    report.save(out / f"{stem}.json", out / f"{stem}.txt")


if __name__ == "__main__":
    main()
