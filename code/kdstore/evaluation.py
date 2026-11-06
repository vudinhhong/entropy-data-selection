"""Exact-match scoring for GSM8K and MATH.

Two metrics are reported throughout, because they disagree in a way that
matters. The lenient metric falls back to the last number in the generation if
no answer marker is found; the strict metric requires the model to produce the
marker. A weakly trained model scores far better on the lenient metric than it
deserves, at a learning rate of 1e-5 the text-only arm scored 37.00 lenient
against 22.52 strict, because it had never learned the `#### N` format and was
being credited for incidental last numbers. Choosing hyper-parameters on the
lenient metric selects exactly those points.
"""
import re
from fractions import Fraction

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
ANSWER_PHRASE = re.compile(r"(?:answer is|answer:)\s*\$?(-?\d[\d,]*\.?\d*)", re.I)
BOXED = re.compile(r"\\boxed\s*\{")

# An un-fine-tuned base model often carries on and invents the next question.
# Cut there before extracting anything, or "the last number" comes from a
# problem the model made up.
NEXT_QUESTION = re.compile(
    r"\n\s*(?:\[?Question\]?\s*[:\]]|Q\s*:|\[?Problem\]?\s*[:\]])", re.I)

_STRIP = ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$", " ",
          "\\text{", "\\mbox{")


def clip(text):
    """Drop anything after the model starts a new question."""
    m = NEXT_QUESTION.search(text)
    return text[:m.start()] if m else text


def last_boxed(text):
    r"""Contents of the final \boxed{...}, counting braces so nesting works."""
    found = None
    for m in BOXED.finditer(text):
        i, depth, out = m.end(), 1, []
        while i < len(text) and depth:
            depth += (text[i] == "{") - (text[i] == "}")
            if depth:
                out.append(text[i])
            i += 1
        if depth == 0:
            found = "".join(out).strip()
    return found


def gold_answer(answer):
    """The reference answer: after #### for GSM8K, inside \\boxed for MATH."""
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    b = last_boxed(answer)
    return b.replace(",", "").strip() if b else None


def predicted_lenient(text):
    """Prefer an explicit marker; fall back to the last number in the text."""
    b = last_boxed(text)
    if b is not None:
        return b.replace(",", "").strip()
    if "####" in text:
        m = NUMBER.search(text.split("####")[-1])
        if m:
            return m.group(0).replace(",", "").rstrip(".")
    m = ANSWER_PHRASE.findall(text)
    if m:
        return m[-1].replace(",", "").rstrip(".")
    m = NUMBER.findall(text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def predicted_strict(text):
    """Require the marker: `#### N` for GSM8K or \\boxed{} for MATH."""
    b = last_boxed(text)
    if b is not None:
        return b.replace(",", "").strip()
    if "####" not in text:
        return None
    m = NUMBER.search(text.split("####")[-1])
    return m.group(0).replace(",", "").rstrip(".") if m else None


def normalise(x):
    r"""Flatten the LaTeX spellings of one value.

    MATH answers are expressions, and the same value has several legal forms
    ($\dfrac12$, $\frac{1}{2}$, $0.5$). Without this, accuracy is understated,
    and understated by different amounts per arm, which bends the comparison
    and not just the absolute level.
    """
    if x is None:
        return None
    x = x.strip().rstrip(".").strip()
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("\\cfrac", "\\frac"), ("^\\circ", ""), ("\\%", ""),
                 ("%", ""), ("\\$", "")):
        x = x.replace(a, b)
    for token in _STRIP:
        x = x.replace(token, "")
    if x.startswith("\\boxed{") and x.endswith("}"):
        x = x[7:-1]
    for prefix in ("x=", "y=", "z=", "n=", "k="):
        if x.startswith(prefix):
            x = x[len(prefix):]
    # strip a wrapping bracket pair: "\left( 3 \right)" is "(3)" by now
    while len(x) > 1 and x[0] in "([" and x[-1] in ")]":
        inner = x[1:-1]
        if (inner.count("(") == inner.count(")")
                and inner.count("[") == inner.count("]")):
            x = inner
        else:
            break
    return x


def _as_fraction(x):
    m = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", x or "")
    if m:
        return Fraction(m.group(1)) / Fraction(m.group(2))
    return Fraction(x)


def equivalent(a, b):
    """Do two answer strings denote the same value?"""
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (ValueError, TypeError):
        pass
    na, nb = normalise(a), normalise(b)
    if na == nb:
        return True
    try:
        return abs(float(_as_fraction(na)) - float(_as_fraction(nb))) < 1e-6
    except Exception:
        return False


def score_generations(generations, golds):
    """Score a list of generated strings against their references."""
    records, lenient, strict = [], 0, 0
    for text, gold in zip(generations, golds):
        text = clip(text)
        reference = gold_answer(gold) if not isinstance(gold, str) or \
            ("####" in gold or "\\boxed" in gold) else gold
        ok = equivalent(predicted_lenient(text), reference)
        st = equivalent(predicted_strict(text), reference)
        lenient += ok
        strict += st
        records.append({"gen": text[:1200], "gold": reference,
                        "ok": bool(ok), "strict": bool(st)})
    n = max(1, len(records))
    return {"n": len(records), "acc": lenient / n, "acc_strict": strict / n,
            "gens": records}


def generate_and_score(model, tokenizer, pairs, device="cuda", batch_size=24,
                       max_new_tokens=320, max_length=None, label="",
                       log=print):
    """Greedy decoding over a test split, scored both ways.

    The batch size is never reduced on an out-of-memory error. Changing the
    batch changes the padding, which changes bf16 arithmetic, which flips
    individual answers: re-running one point at a different batch size left
    aggregate accuracy within 0.30 points but moved 86 of 1,319 answers. Wait
    for memory instead of silently measuring something else.
    """
    import math

    import torch

    from .models import MAX_LENGTH, PROMPT

    max_length = max_length or MAX_LENGTH
    tokenizer.padding_side = "left"
    model.eval()

    generations, golds = [], []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            enc = tokenizer([PROMPT.format(q=q) for q, _ in chunk],
                            return_tensors="pt", padding=True, truncation=True,
                            max_length=max_length,
                            add_special_tokens=False).to(device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
            generations += tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            golds += [a for _, a in chunk]
            if (start // batch_size) % 10 == 0:
                log(f"  {label} eval {len(generations)}/{len(pairs)}")

    scored = score_generations(generations, golds)
    n = max(1, scored["n"])
    scored["se"] = math.sqrt(scored["acc"] * (1 - scored["acc"]) / n)
    return scored
