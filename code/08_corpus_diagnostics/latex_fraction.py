"""Share of answer tokens that carry LaTeX markup, GSM8K vs MATH."""
from datasets import load_dataset
from transformers import AutoTokenizer
LATEX = set("\\${}^_&")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")

def frac(answers, name):
    total = marked = 0
    for a in answers:
        ids = tok(a, add_special_tokens=False)["input_ids"]
        total += len(ids)
        for t in ids:
            if any(c in LATEX for c in tok.decode([t])):
                marked += 1
    print(f"  {name:8s} answer tokens {total:8d}   LaTeX-bearing {marked:7d}  = {100*marked/total:.1f}%")

g = load_dataset("openai/gsm8k", "main")["train"]
frac([r["answer"] for r in g], "GSM8K")

tr = []
for cfg in ("algebra","counting_and_probability","geometry","intermediate_algebra",
            "number_theory","prealgebra","precalculus"):
    tr += [r["solution"] for r in load_dataset("EleutherAI/hendrycks_math", cfg)["train"]]
frac(tr, "MATH")
