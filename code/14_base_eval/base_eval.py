"""Untrained-base accuracy, one output file per case.

Cases are given on the command line. Each result is written to its own file,
named after the corpus, the model and the generation budget, so runs that
differ in any of the three do not overwrite one another.

    python code/14_base_eval/base_eval.py math:Qwen/Qwen1.5-1.8B:320
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "code")
from kdstore import load_pairs                                  # noqa: E402
from kdstore.evaluation import generate_and_score               # noqa: E402
from kdstore.models import load_model, load_tokenizer           # noqa: E402

OUT = Path("results/base_accuracy")
KNOWN = {            # what the paper reports, for an immediate comparison
    "math|Qwen/Qwen1.5-1.8B": 7.60,
    "gsm8k|Qwen/Qwen1.5-1.8B": 24.49,
    "math|Qwen/Qwen2-1.5B": 11.40,
    "ecqa|Qwen/Qwen1.5-1.8B": 43.80,
}


def filename(dataset, model_id, max_new):
    """A name that differs whenever the run differs."""
    model = model_id.replace("/", "-")
    return f"{dataset}_{model}_max{max_new}.json"


def main():
    cases = []
    for arg in sys.argv[1:]:
        parts = arg.split(":")
        if len(parts) < 3:
            raise SystemExit(f"expected corpus:model:max_new_tokens, "
                             f"got {arg!r}")
        cases.append((parts[0], ":".join(parts[1:-1]), int(parts[-1])))
    if not cases:
        raise SystemExit(__doc__)

    OUT.mkdir(parents=True, exist_ok=True)
    for dataset, model_id, max_new in cases:
        path = OUT / filename(dataset, model_id, max_new)
        if path.exists():
            # Greedy decoding on a fixed split is deterministic, so there is
            # nothing to gain from repeating a case, and overwriting is the
            # failure this script exists to avoid.
            got = json.loads(path.read_text())
            print(f"{path}: already measured, {got['acc']:.2f}", flush=True)
            continue
        pairs = load_pairs(dataset)["test"]
        tok = load_tokenizer(model_id, padding_side="left")
        model = load_model(model_id, train=False, device="cuda")
        res = generate_and_score(model, tok, pairs, device="cuda",
                                 batch_size=16, max_new_tokens=max_new,
                                 label=f"{dataset}-base")
        acc = 100 * res["acc"]
        path.write_text(json.dumps({
            "corpus": dataset, "model": model_id, "max_new_tokens": max_new,
            "n": res["n"], "acc": acc,
            "acc_strict": 100 * res["acc_strict"],
        }, indent=2))
        want = KNOWN.get(f"{dataset}|{model_id}")
        verdict = ""
        if want is not None:
            verdict = ("  MATCHES the paper" if abs(acc - want) < 0.05
                       else f"  <-- the paper reports {want}")
        print(f"{path}: lenient {acc:.2f}  strict "
              f"{100 * res['acc_strict']:.2f}  (n={res['n']}){verdict}",
              flush=True)
        del model

    print(f"\n{OUT} now holds:")
    for p in sorted(OUT.glob("*.json")):
        d = json.loads(p.read_text())
        print(f"  {p.name:52s} {d['acc']:6.2f}")


if __name__ == "__main__":
    main()
