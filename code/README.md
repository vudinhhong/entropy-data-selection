# Code

Seventeen numbered directories and one shared library. Each numbered directory
holds one script and, where it needs one, a `config.json`; the script runs
every experiment the config names, prints its tables to the terminal and
writes them to `results/`. Every path in a config is relative to the
repository root, so the configs work wherever the repository is cloned.

    kdstore/                  shared library: data loading, gating, selection,
                              training, evaluation, paired statistics

    01_build_store/           corpus split, teacher fine-tune, entropy store,
                              the fold subsets every arm selects from      GPU
    02_train_students/        every training arm in the paper              GPU
    03_store_analysis/        store size and censoring                     GPU
    04_calibration/           base-model calibration                       GPU
    05_results/               tables, rebuilt from saved run output        CPU
    06_baselines/             the sequence-level and text-only baselines   GPU
    07_figures/               Fig 1                                        CPU
    08_corpus_diagnostics/    per-corpus gate thresholds and entropies     CPU
    09_selection_baselines/   IFD and Superfiltering, as published         GPU
    10_truncation/            the top-K check behind Fig 1C                CPU
    11_s1_table/              the paired-t and McNemar readings side by side
    12_criterion_vs_difficulty/  the criterion against four students' IFD
    13_whymath_check/         the corpus-property comparison
    14_base_eval/             untrained base accuracy per cell             GPU
    15_disagreement/          teacher-vs-reference disagreement per corpus GPU
    16_contrast/              the two-ended contrast, the paper's main test
    17_reproduce/             verification against data/per_run_accuracies.csv

    examples/                 the shortest path from a store to a batch
    tests/                    unit tests for the library

## Without a GPU

The per-run accuracies are released in `data/per_run_accuracies.csv`, so every
statistical claim rebuilds without training anything:

    python code/17_reproduce/verify_paper_numbers.py   # stdlib only
    python code/16_contrast/two_ended_contrast.py      # stdlib only
    python code/07_figures/make_figures.py             # needs matplotlib

## With a GPU

Order matters only once: the store and the fold subsets have to exist before
any student trains.

    python code/01_build_store/build_store.py --stage teacher --only gsm8k
    python code/01_build_store/build_store.py --stage fold_selection --only gsm8k
    python code/02_train_students/run_experiments.py --group fold_selection_q15_18

`--list` shows the groups and how many points each holds, `--dry-run` prints
the plan without training, and `--shard i/n` splits a group across cards.
The runner skips points whose results already exist, so an interrupted sweep
resumes by being started again.

## Group names

The experiment groups the paper's tables are computed from:

    fold_selection_q15_18         GSM8K      / Qwen1.5-1.8B   (the anchor cell)
    fold_selection_math2_q2_15    MATH       / Qwen2-1.5B
    fold_ecqa_q15_18              ECQA       / Qwen1.5-1.8B
    fold_orca_anchor_half         Orca-Math  / Qwen1.5-1.8B
    fold_ape210k_ends             APE210K    / Qwen1.5-1.8B
    fold50_llama_gsm8k            GSM8K      / Llama-3.2-1B
    fold50_smol_gsm8k             GSM8K      / SmolLM2-360M

## Output naming

Every script names its output file after the configuration that produced it,
and prints the full path plus a warning when it replaces an existing file.
Two runs that differ in any setting therefore write to two files.
