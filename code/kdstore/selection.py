"""Choosing which sequences go into the artifact.

Two layers, and only the second one looks at the teacher.

Layer 1 partitions the pool by a structural property of the answer, the
number of reasoning steps, and gives each stratum a quota proportional to its
size. This keeps the shape of the pool, so that any difference against a random
baseline cannot be a distribution shift.

Layer 2 ranks within each stratum. `random` is the control, and it is what the
paper's released subsets use. The other scores are derived from the teacher's
distribution, which makes layer 2 the only place sequence selection can use
information the plain corpus does not carry.

A warning about the scores that sum over positions instead of averaging:
`H_total` and `gated_count` both grow with sequence length, so selecting by
them buys more tokens as well as better ones. At a fixed budget of sequences
that confounds "chose better" with "chose more", and the comparison has to be
read at matched token count, not matched sequence count. `summary()` reports
both so the confound is printed explicitly.
"""
from pathlib import Path

import numpy as np

STEP_BINS = [0, 2, 3, 4, 5, 6, 8, 10 ** 9]


def step_ladder(records, min_share=0.015, key="steps"):
    """Derive stratum edges from the pool's own distribution of step counts.

    The fixed ladder above was set for GSM8K, where answers cluster around
    three or four steps and the open top bin catches almost nothing. On a
    corpus with a long tail it catches a great deal, and then equal quotas
    stop meaning equal composition: two arms can take the same number of
    sequences from the top stratum and still differ by two steps on average,
    because one of them prefers the longest members of an unbounded bin. That
    is a difficulty confound wearing the costume of a controlled comparison.

    The rule here walks the distinct step counts in order and closes a
    stratum once it holds `min_share` of the pool, so no stratum is wider
    than the data require. Callers should check the result: `selection_summary`
    reports mean steps per arm, and if two arms drawn with equal quotas still
    differ, the ladder is too coarse for that pool.
    """
    counts = {}
    for r in records:
        if r.get("npos", 0) > 0:
            counts[r[key]] = counts.get(r[key], 0) + 1
    total = sum(counts.values())
    edges, run = [0], 0
    for value in sorted(counts):
        run += counts[value]
        if run >= min_share * total:
            edges.append(value + 1)
            run = 0
    if run:                       # the leftover tail joins the last stratum
        edges[-1] = 10 ** 9
    else:
        edges.append(10 ** 9)
    return edges


def stratum_of(record, bins=None, key="steps"):
    """Which stratum a record falls in.

    `key` is the structural property layer 1 partitions on. It is the number
    of lines in the reference answer wherever that is informative, which on
    GSM8K and APE210K it is. On a corpus whose answers are one or two lines
    regardless of difficulty -- ECQA, where 64% of them have the same line
    count, that proxy is degenerate and the answer's token count is used
    instead. Either way it is a property of the text, available without a
    model, which is what layer 1 is for.
    """
    bins = bins or STEP_BINS
    steps = record[key]
    for b in range(len(bins) - 1):
        if bins[b] <= steps < bins[b + 1]:
            return b
    return len(bins) - 2


# Each score returns a value to be sorted ASCENDING, so smaller is picked first.
SCORES = {
    "random": None,
    "H_low": lambda r: +r["mean_H"],          # least uncertain on average
    "H_high": lambda r: -r["mean_H"],         # most uncertain on average
    "nll_low": lambda r: +r["mean_nll"],      # teacher surest of the gold answer
    "nll_high": lambda r: -r["mean_nll"],     # hardest for the teacher, LESS-like
    "H_total": lambda r: -r["mean_H"] * r["npos"],   # most entropy in total
    "gated_count": lambda r: -r.get("gated", 0),     # most positions past the gate
    "gated_H": lambda r: -r.get("gated_H", 0.0),     # most entropy past the gate
    # The other end of the same scale. If selection mattered at all, the
    # sequences the gate has least to say about should train a worse
    # student than a random draw; if they do not, the null is a null and
    # not a failure of power.
    "gated_H_low": lambda r: +r.get("gated_H", 0.0),  # least entropy past the gate
    "length": lambda r: -r["npos"],           # longest, ignoring the teacher
    # Published criteria, read from a file an external scorer wrote. IFD keeps
    # the samples whose answer the question helps least with; its authors cap
    # the score at one, which `annotate_scores` applies when it loads them.
    "ifd_high": lambda r: -r.get("ifd_high", 0.0),
    # Superfiltering is the same ratio scored by a small proxy model, which is
    # the whole of its proposal, so it enters as a separate arm instead of as
    # a setting of the one above.
    "superfilter_high": lambda r: -r.get("superfilter_high", 0.0),
}


def annotate_scores(records, path, key, cap=None):
    """Attach a score computed outside this package, by corpus index.

    `cap` drops records above a ceiling instead of ranking them, which is how
    IFD is used in the paper that proposes it: a ratio above one means the
    answer is easier with the question removed, which the authors read as a
    broken sample instead of an informative one.
    """
    import json
    payload = json.loads(Path(path).read_text())["scores"]
    kept = 0
    for r in records:
        entry = payload.get(str(r["i"]))
        if entry is None:
            r[key] = 0.0
            continue
        value = entry.get(key, entry.get("ifd"))
        if cap is not None and value >= cap:
            r[key] = 0.0
            continue
        r[key] = float(value)
        kept += 1
    return kept


def annotate_gate(records, store, tau):
    """Add per-sequence gate statistics, which the store metadata lacks."""
    offsets = store.array("offsets")
    H = store.entropy()
    for r in records:
        lo, hi = int(offsets[r["i"]]), int(offsets[r["i"] + 1])
        h = H[lo:hi]
        passed = h > tau
        r["gated"] = int(passed.sum())
        r["gated_H"] = float(h[passed].sum())
    return records


def select_subset(records, fraction, score="random", seed=1234, partition=True,
           pool=None, token_budget=None, bins=None, stratify_by="steps"):
    """Pick a subset. Returns (indices, info).

    `token_budget` caps the number of answer positions instead of the number
    of sequences, which is how to compare a length-sensitive score against a
    length-blind one without handing the former extra data.
    """
    rng = np.random.default_rng(seed)
    rows = [r for r in records if r.get("npos", 0) > 0]
    if pool is not None:
        allowed = set(int(i) for i in pool)
        rows = [r for r in rows if r["i"] in allowed]
    if not rows:
        raise ValueError("the pool does not intersect the store")

    n = len(rows)
    budget = max(1, int(round(fraction * n)))

    if not partition:
        picked = rng.choice([r["i"] for r in rows], size=budget, replace=False)
        chosen = sorted(int(i) for i in picked)
        return chosen, {"mode": "global_random", "budget": budget, "n_pool": n}

    groups = {}
    for r in rows:
        groups.setdefault(stratum_of(r, bins, stratify_by), []).append(r)

    keys = sorted(groups)
    exact = {k: budget * len(groups[k]) / n for k in keys}
    quota = {k: int(np.floor(exact[k])) for k in keys}
    for k in sorted(keys, key=lambda k: -(exact[k] - quota[k]))[:budget - sum(quota.values())]:
        quota[k] += 1

    # A token budget is spent per stratum, in proportion to stratum size, so
    # layer 1 still shapes the sample while layer 2 does the choosing. Spent
    # globally it would fill from whichever stratum came first and leave the
    # score nothing to rank. Within a stratum we walk the ranking and stop at
    # the first sequence that will not fit, instead of skipping it and
    # carrying on: skipping quietly prefers short sequences, which is its own
    # selection rule and not the one under test.
    token_quota = ({k: token_budget * len(groups[k]) / n for k in keys}
                   if token_budget is not None else None)

    chosen, strata = [], []
    for k in keys:
        group = groups[k]
        if score == "random":
            order = [int(j) for j in rng.permutation(len(group))]
        else:
            f = SCORES[score]
            values = [f(r) for r in group]
            if len(group) > 1 and len(set(values)) == 1:
                # Every record scored the same, so the ranking is whatever
                # order the pool happened to be in. That is a bug in the
                # caller, usually a score field that was never populated --
                # and it produces a subset that looks plausible and means
                # nothing, so refuse it instead of return it.
                raise ValueError(
                    f"score {score!r} is constant over {len(group)} records; "
                    "the field it reads was probably never annotated")
            order = [int(j) for j in np.argsort(values, kind="stable")]

        if token_quota is None:
            taken = [group[j]["i"] for j in order[:min(quota[k], len(group))]]
        else:
            taken, spent = [], 0
            for j in order:
                cost = group[j]["npos"]
                if spent + cost > token_quota[k]:
                    break
                taken.append(group[j]["i"])
                spent += cost

        chosen += taken
        edges = bins or STEP_BINS
        strata.append({"stratum": k, "range": [edges[k], edges[k + 1]],
                       "size": len(group), "taken": len(taken)})

    chosen = sorted(int(i) for i in chosen)

    return chosen, {"mode": f"partition_steps+{score}", "budget": budget,
                    "n_pool": n, "strata": strata,
                    "token_budget": token_budget}


def selection_summary(indices, records):
    """What a selection actually bought, in sequences, tokens and gated slots."""
    # Sequences with no answer positions carry only an index in the store's
    # metadata, so skip them instead of reading fields they do not have.
    by_index = {r["i"]: r for r in records if r.get("npos", 0) > 0}
    rows = [by_index[i] for i in indices if i in by_index]
    if not rows:
        return {"sequences": 0, "positions": 0, "positions_per_sequence": 0.0,
                "mean_H": 0.0, "total_H": 0.0, "gated_positions": 0,
                "mean_steps": 0.0}
    total_positions = sum(r["npos"] for r in rows)
    return {
        "sequences": len(rows),
        "positions": total_positions,
        "positions_per_sequence": total_positions / max(1, len(rows)),
        "mean_H": float(np.mean([r["mean_H"] for r in rows])),
        "total_H": float(sum(r["mean_H"] * r["npos"] for r in rows)),
        "gated_positions": int(sum(r.get("gated", 0) for r in rows)),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
    }
