"""The corpora, reduced to (question, answer) pairs.

GSM8K arrives as a single config. MATH has to be assembled: the training half
is the union of the seven Hendrycks subject configs, while evaluation uses
MATH-500, the 500-problem subset the community has settled on. Using MATH-500
keeps our accuracies comparable with published ones and costs a tenth of what
the full 5,000-problem test set costs to generate.

Two further corpora were added to test the scope condition the GSM8K and MATH
comparison suggests, namely that the criterion needs a teacher able to solve
the corpus. APE210K is elementary arithmetic, where a 4B teacher should be
comfortable; ECQA is commonsense multiple choice, which is not mathematics at
all. Every corpus is rewritten into the same `#### answer` convention GSM8K
uses, so that one scorer serves all of them and no corpus gets a metric of its
own.
"""

MATH_CONFIGS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)


def load_pairs(dataset="gsm8k"):
    """Return {"train": [(question, answer), ...], "test": [...]}."""
    from datasets import load_dataset

    name = dataset.lower()
    if name == "gsm8k":
        d = load_dataset("openai/gsm8k", "main")
        return {
            split: [(r["question"], r["answer"]) for r in d[split]]
            for split in ("train", "test")
        }

    if name == "orca":
        # Grade-school word problems written by GPT-4 from a seed set. Chosen
        # because it is not derived from GSM8K and because 200k rows is large
        # enough that a tenth of the student half is still enough data for a
        # small model to saturate on, the condition GSM8K cannot meet, since
        # a tenth of its student half is 374 sequences.
        import re
        d = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
        pairs = []
        for r in d:
            nums = re.findall(r"-?\d[\d,]*\.?\d*", r["answer"].replace("$", ""))
            if not nums:                       # 576 of 200,035 carry no number
                continue
            final = nums[-1].replace(",", "").rstrip(".")
            pairs.append((r["question"].strip(),
                          r["answer"].strip() + "\n#### " + final))
        # The corpus ships no test split, so one is carved off the end. The
        # order is the file's own, which is not sorted by anything we select
        # on, and the cut is deterministic.
        return {"train": pairs[:-2000], "test": pairs[-2000:]}

    if name == "math":
        train = []
        for config in MATH_CONFIGS:
            d = load_dataset("EleutherAI/hendrycks_math", config)
            train += [(r["problem"], r["solution"]) for r in d["train"]]
        t = load_dataset("HuggingFaceH4/MATH-500")
        split = "test" if "test" in t else next(iter(t))
        test = [(r["problem"], r.get("solution") or r["answer"]) for r in t[split]]
        return {"train": train, "test": test}

    if name == "ape210k":
        # 195k problems is far more than the experiment needs, and an unequal
        # corpus size would confound every comparison against GSM8K, so we
        # take a seeded sample of exactly GSM8K's training size.
        import random
        d = load_dataset("MU-NLPC/calc-ape210k")
        rows = [r for r in d["train"] if r["chain"].count("\n") >= 2]
        random.Random(1234).shuffle(rows)
        train = [(r["question"], _ape_answer(r)) for r in rows[:7473]]
        test = [(r["question"], _ape_answer(r)) for r in d["test"]]
        return {"train": train, "test": test}

    if name == "ecqa":
        d = load_dataset("yangdong/ecqa")
        return {split: [_ecqa_pair(r) for r in d[split]]
                for split in ("train", "test")}

    if name == "kk":
        import pandas as pd
        base = ("https://huggingface.co/datasets/K-and-K/"
                "knights-and-knaves/resolve/main")
        def read(split, sizes):
            frames = [pd.read_json(f"{base}/{split}/people{n}_num{c}.jsonl",
                                   lines=True) for n, c in sizes]
            return pd.concat(frames, ignore_index=True)
        train = read("train", [(2, 200)] + [(n, 1000) for n in range(3, 9)])
        test = read("test", [(n, 100) for n in range(2, 9)])
        return {"train": [_kk_pair(r) for _, r in train.iterrows()],
                "test": [_kk_pair(r) for _, r in test.iterrows()]}

    raise ValueError(f"unknown dataset: {dataset!r} (expected 'gsm8k', "
                     "'math', 'ape210k', 'ecqa' or 'kk')")


def _kk_pair(row):
    """Knights and Knaves: a deductive chain and a per-person verdict.

    The reference chain is one line per assumption and one per contradiction,
    which is what the first selection layer partitions on. The verdict is a
    list of roles, and we write it as a digit string -- 1 for knight, 2 for
    knave, in the order the puzzle introduces the inhabitants, so that the
    same numeric scorer serves this corpus as serves the others. Zero is
    avoided because a leading zero does not survive the extractor.
    """
    steps = [str(x) for x in row["cot_repeat_steps"]]
    chain = "\n".join([str(row["cot_head"])] + steps + [str(row["cot_foot"])])
    verdict = "".join("1" if bool(v) else "2" for v in row["solution"])
    return str(row["quiz"]), f"{chain}\n#### {verdict}"


def _ape_answer(row):
    """APE210K ships a calculator chain; keep it and add GSM8K's marker."""
    chain = row["chain"].split("<result>")[0].strip()
    return f"{chain}\n#### {row['result_float']}"


def _ecqa_pair(row):
    """ECQA as a numbered multiple choice with its human justification.

    The options are numbered and the answer is the index, so the existing
    numeric scorer works unchanged and no corpus needs a scorer of its own.
    The reference reasoning is the positive justification, which is written
    one claim per line; `taskB`, the free-flow version, is a single paragraph
    91% of the time and would leave the first selection layer with nothing to
    stratify on.
    """
    options = [row[f"q_op{k}"] for k in range(1, 6)]
    listed = "\n".join(f"{k}. {opt}" for k, opt in enumerate(options, 1))
    question = f"{row['q_text']}\n{listed}"
    index = options.index(row["q_ans"]) + 1
    justification = str(row["taskA_pos"]).strip()
    return question, f"{justification}\n#### {index}"


def load_written_pairs(path):
    """Read a corpus the teacher wrote, as produced by 06_baselines.

    Sequence-level distillation replaces the reference answers with the
    teacher's own; the file holds the prompts it was given and the text it
    produced. Anything downstream that wants to treat that text as a corpus --
    building a store over it, training on it, reads it through here, so the
    two stay in step.
    """
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text())
    return [tuple(x) for x in payload["pairs"]]
