"""Teacher accuracy on each corpus.

The teacher generates an answer for every prompt in a subset of the student
half, text it was not fine-tuned on, and the answers are scored with the same
extractor used for the students.

    python code/09_selection_baselines/measure_teacher.py --only ecqa
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from kdstore import Report, load_config, load_pairs                 # noqa: E402
from kdstore.evaluation import score_generations                    # noqa: E402
from kdstore.models import load_model, load_tokenizer               # noqa: E402
from kdstore.paths import resolve                                   # noqa: E402


@torch.no_grad()
def generate(teacher, tok, prompts, batch_size, max_new):
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=768).to(teacher.device)
        ids = teacher.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            out.append(tok.decode(ids[j][enc.input_ids.shape[1]:],
                                  skip_special_tokens=True))
        if i % (batch_size * 10) == 0:
            print(f"    {i}/{len(prompts)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "teachers.json"))
    ap.add_argument("--only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = Report("Teacher competence on each corpus")
    rows = []
    for spec in cfg["experiments"]:
        if args.only and spec["name"] != args.only:
            continue
        corpus = load_pairs(spec["dataset"])["train"]
        idx = json.loads(resolve(spec["subset"]).read_text())["idx"]
        tok = load_tokenizer(spec["tokenizer"], padding_side="left")
        teacher = load_model(resolve(spec["teacher"]), train=False,
                             device=spec.get("device", "cuda"))
        texts = generate(teacher, tok, [corpus[i][0] for i in idx],
                         spec.get("batch_size", 16),
                         spec.get("max_new_tokens", 320))
        scored = score_generations(texts, [corpus[i][1] for i in idx])
        rows.append([spec["name"], spec["dataset"], scored["n"],
                     round(100 * scored["acc"], 1),
                     round(100 * scored["acc_strict"], 1)])
        print(f"  {spec['name']}: {100 * scored['acc']:.1f}% lenient, "
              f"{100 * scored['acc_strict']:.1f}% strict", flush=True)
        del teacher
        torch.cuda.empty_cache()

    report.table("teacher_competence",
                 ["teacher", "corpus", "prompts", "solved (%)",
                  "strict (%)"], rows,
                 title="Teacher accuracy on the half the students train on")
    out = resolve(cfg.get("report_dir", "results"))
    # Name the report after the corpora measured, so a later run with --only
    # does not replace the record of the others.
    stem = "teacher_competence"
    if args.only:
        stem += "_" + args.only
    else:
        stem += "_" + "_".join(e["name"] for e in cfg["experiments"])
    report.save(out / f"{stem}.json", out / f"{stem}.txt")


if __name__ == "__main__":
    main()
