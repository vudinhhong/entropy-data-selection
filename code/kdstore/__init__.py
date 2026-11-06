"""Entropy-gated top-K logit stores: library code shared by the experiments.

    LogitStore          read a published store, apply the gate, slice to K
    LogitStoreDataset   pair the store with the corpus for training
    censored_kd_loss    the K+1-bucket objective
    load_pairs          GSM8K and MATH as (question, answer) pairs

See `examples/load_artifact.py` for the shortest path from a downloaded store
to a training batch.
"""
from .data import load_pairs, load_written_pairs
from .evaluation import generate_and_score, score_generations
from .dataset import (LogitStoreDataset, build_dataloader, collate_fn,
                      gate_mask)
from .losses import censored_kd_loss
from .reporting import Report, load_config, required
from .selection import (annotate_gate, annotate_scores, select_subset,
                        selection_summary)
from .store import LogitStore, binary_entropy, storage_cost

__all__ = [
    "LogitStore", "LogitStoreDataset", "build_dataloader", "collate_fn",
    "gate_mask", "censored_kd_loss", "load_pairs", "Report", "load_config",
    "required",
    "generate_and_score", "score_generations", "load_written_pairs",
    "binary_entropy", "storage_cost",
    "select_subset", "selection_summary", "annotate_gate",
]
