# Selecting fine-tuning data by a teacher's predictive entropy

Code, selections and per-run results for the paper *Corpus-dependent direction of teacher predictive entropy in fine-tuning
data selection* (Vu, Duong, Le and Le).

Score each training sequence by the entropy a teacher carries at the positions
where that entropy passes a threshold, then keep the top or bottom slice.
One teacher-forced pass, no gradients, no second model. The headline finding
is that the score separates the two ends of its own scale on four of five
corpora — and that **which end is the good one changes between corpora**, with
nothing we measured predicting it.

## What is here

```
code/        every script behind the paper, plus the shared library
data/        per-run accuracies, the selected index lists, the gate
             thresholds and the teacher generations
figures/     Fig 1 and S1 Table, built by the scripts from the data below
```

`data/per_run_accuracies.csv` is the minimal dataset: every paired test in the
paper is a t-test on fold-level differences computed from these rows and
nothing else. 1,631 runs, one per row. Two columns say how each run relates to
the paper:

| `status` | meaning |
|---|---|
| `reported` | the run is behind a number in the paper; `used_for` says which |
| `superseded` | replaced by a later run on the same folds, named in the Limitations |
| `incomplete` | the group was stopped before all folds finished |
| `other-experiment` | a stored-distribution experiment, outside the scope of this paper |

`data/subsets/` holds the selected index lists — integer indices into the
public corpora, which is the criterion's actual output and the thing a reader
would reuse. It covers GSM8K, MATH, ECQA, Orca-Math and APE210K;
`data/subsets_index.json` summarises them.
`data/gate_thresholds.json` gives the gate per corpus and teacher, with the
coverage target each was calibrated to.
`data/teacher_generations/` holds the solutions the teacher wrote for GSM8K
and MATH, from which those two competence figures are recomputed.

Not here, and named in the paper as such: the teacher stores (hundreds of
megabytes of top-K log-probabilities per corpus), the model checkpoints (LoRA
adapters over public bases), the index lists for the untrained-teacher and
Llama-teacher cells, and the teacher generations for ECQA, APE210K and the
Llama teacher. The last two groups were produced on a cluster whose working
directories have since been cleared. Stores and index lists are rebuilt by
`code/01_build_store/`, teacher generations by
`code/09_selection_baselines/measure_teacher.py` (its `teachers.json` already
holds the three cells), in both cases from the public datasets and the
released configuration, which fixes every threshold and seed.

## Verifying the paper's numbers (CPU only)

```bash
python code/17_reproduce/verify_paper_numbers.py
```

Standard library only. It recomputes twenty of the paper's headline values
and the mean, median, worst and lost-cell columns of all seven rows of the
direction-pilot table, and prints each beside the value the paper prints, so
any disagreement is visible. All of them agree.

```bash
python code/18_direction/direction_pilot.py     # PDS, Table 9
```

Also standard library only. It replays Algorithm 1, Pilot-Based Direction Selection, over every
cell that ran both ends of the scale: the pilot sees `m` of the five folds,
picks a direction from those alone, and is scored on the folds it did not
see. `--strict` scores with strict accuracy instead of lenient; the ordering
of the policies does not change.

## Running the pilot on a corpus of your own

```bash
python code/18_direction/run_pilot.py \
    --pairs train.jsonl --eval-pairs test.jsonl --out pilot/ \
    --teacher Qwen/Qwen3-4B-Base --student Qwen/Qwen1.5-1.8B \
    --budget 75 --folds 2
```

This is Algorithm 1 of the paper, Pilot-Based Direction Selection, end to
end: one teacher forward pass, two
folds, the top and bottom of each at your budget, four fine-tunes, and the
direction with the bound on what reading it wrongly can cost. It writes
`pilot/direction.json`. Add `--plan` to write the configuration and print the
commands without running anything, which needs no GPU and no model download.

Input is JSONL with a `question` and an `answer` per line. The budget should
be the one you will train at: the paper shows the direction transferring
across a five-fold change of budget on one cell and establishes no more than
that.

The script writes the configuration that `01_build_store` and
`02_train_students` read and then runs them, so the pilot and the paper go
through the same code. `python code/tests/test_pilot_config.py` checks that
the three scripts still agree on the filenames they pass between them; it is
standard library only and runs in `run_all.sh check`.

The pipeline has been run end to end on a corpus supplied this way: 600 GSM8K
sequences, a Qwen2.5-0.5B teacher and student, a budget of 50 and two folds,
which is six and a half minutes on one RTX 4090. The gate came out at
`tau = 0.5032` nats for the 29.7% coverage asked for, far from the 0.0335 a
4B teacher gives on the same corpus, which is why the threshold is set by
coverage and not in nats.

One thing it cannot choose for you is the metric. Students are scored by the
exact-match rule in `code/kdstore/evaluation.py`, written for answers ending
in a number or a `\boxed{}` expression. A corpus whose answers look different
needs its own scorer, and the comparison is worth no more than that scorer is.

```bash
pip install -r requirements.txt
python code/16_contrast/two_ended_contrast.py   # reproduces the main table
python code/07_figures/make_figures.py          # rebuilds figures/fig1.pdf
bash code/run_all.sh check                      # the first two together
```

## Re-run the experiments, with a GPU

See `code/README.md` for the directory map and the group names. In short:

```bash
pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt

python code/01_build_store/build_store.py --stage teacher --only gsm8k
python code/01_build_store/build_store.py --stage fold_selection --only gsm8k
python code/02_train_students/run_experiments.py --group fold_selection_q15_18
```

The runner skips points whose results already exist, so an interrupted sweep
resumes by being started again; `--shard i/n` splits a group across cards,
`--list` shows the groups and `--dry-run` prints the plan. On one RTX 4090 a
1.5–1.8B student at the 75-sequence budget trains and evaluates in about
fifteen minutes and peaks at 16.4 GB of VRAM; the Orca-Math arms train on
14,808 sequences and take about two hours each.

## Notes on interpretation

**Run-to-run variance.** Two runs on the same subset differing
only in seed land 0.95 to 1.33 points apart, which is the size of the effects
the paper reports. Every number here is a two-seed mean paired at the fold.

**Detection floors.** Each comparison has a detection floor,
`t(0.975, 4) · s / sqrt(5)`, printed by the verification script. On APE210K
that floor is above two points because the budget is 75 sequences, so its
comparisons against a random control cannot resolve a one-point effect no
matter what is true.

**Selection bias in the per-corpus table.** The table of the
better end against a random control carries a selection bias, which the paper
states rather than corrects. The unbiased test is the two-ended contrast in
`code/16_contrast/`, which chooses no end.

## Regenerating the run table

`data/per_run_accuracies.csv` is built from the raw per-run JSON, which also
holds the per-item generations and is far too large to release:

```bash
python code/17_reproduce/build_run_table.py results/grid
python code/17_reproduce/verify_paper_numbers.py
```

The first directory given wins a tag clash, so the authoritative results
directory goes first.

One script cannot be re-run from the archive. `code/11_s1_table/make_s1.py`
computes a McNemar test, which needs the per-item correctness vectors rather
than per-run accuracies; those are not released, so its output ships ready
built as `figures/S1_Table.tex` and `figures/S1_Table.pdf`.

The `status` and `used_for` columns are added by hand
afterwards; the script does not infer them.

## Use of generative AI tools

Claude (Anthropic, 2025–2026) was used to assist with the experiment scripts
and the analysis code in this repository, and ChatGPT (GPT-4o, OpenAI) for
English language editing of the accompanying paper. The study design, the
choice of experiments and the interpretation of every result are the authors'.
Every script here was read and tested by the authors, every number in the
paper is reproduced from
`data/per_run_accuracies.csv` by `code/17_reproduce/verify_paper_numbers.py`,
and the authors take full responsibility for the content.

## Citing

Archive DOI: 10.5281/zenodo.XXXXXXX (filled in when the record is published).

See `CITATION.cff`. `code/` and `figures/` are MIT; `data/` is CC BY 4.0. The
public datasets and models keep their own licences — see `LICENSE`, and note
that the two Llama models are under the Llama 3.2 Community Licence rather
than an open-source licence.
