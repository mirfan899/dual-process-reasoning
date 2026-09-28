"""Predictive Shannon entropy and uncertainty metric modules."""

import numpy as np
from typing import List, Sequence
from dataclasses import dataclass


@dataclass
class StepEntropyStats:
    mean_entropy: float
    max_entropy: float
    top_decile_entropy: float
    token_count: int


def compute_shannon_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Computes Shannon entropy in bits for token distribution."""
    clipped = np.clip(probs, eps, 1.0)
    clipped = clipped / np.sum(clipped)
    return float(-np.sum(clipped * np.log2(clipped)))


def compute_step_entropy(token_entropies: Sequence[float], top_decile_ratio: float = 0.10) -> StepEntropyStats:
    """Isolates the highest uncertainty token transitions in a reasoning step."""
    if not token_entropies:
        return StepEntropyStats(0.0, 0.0, 0.0, 0)
    arr = np.array(token_entropies, dtype=np.float64)
    k_top = max(1, int(np.ceil(len(arr) * top_decile_ratio)))
    sorted_h = np.sort(arr)[::-1]
    return StepEntropyStats(
        mean_entropy=float(np.mean(arr)),
        max_entropy=float(np.max(arr)),
        top_decile_entropy=float(np.mean(sorted_h[:k_top])),
        token_count=len(arr)
    )
