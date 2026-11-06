"""8-gram overlap between test questions and training questions."""
import re, sys
from datasets import load_dataset

def grams(text, n=8):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i+n]) for i in range(max(0, len(w)-n+1))}

def report(name, train_q, test_q):
    pool = set()
    per = []
    for q in train_q:
        g = grams(q); per.append(g); pool |= g
    hits_union = 0; hits_single = 0
    for q in test_q:
        g = grams(q)
        if not g: continue
        if len(g & pool)/len(g) > 0.5:
            hits_union += 1
            if any(len(g & t)/len(g) > 0.5 for t in per): hits_single += 1
    n = len(test_q)
    print(f"{name}: train {len(train_q)}, test {n}")
    print(f"   >50% of 8-grams in the union of training questions: {hits_union}  ({100*hits_union/n:.2f}%)")
    print(f"   >50% shared with a single training question:        {hits_single}  ({100*hits_single/n:.2f}%)")

g = load_dataset("openai/gsm8k", "main")
report("GSM8K", [r["question"] for r in g["train"]], [r["question"] for r in g["test"]])

m = load_dataset("EleutherAI/hendrycks_math", "algebra")
tr = []
for cfg in ("algebra","counting_and_probability","geometry","intermediate_algebra",
            "number_theory","prealgebra","precalculus"):
    d = load_dataset("EleutherAI/hendrycks_math", cfg)
    tr += [r["problem"] for r in d["train"]]
m500 = load_dataset("HuggingFaceH4/MATH-500")
report("MATH-500", tr, [r["problem"] for r in m500["test"]])
