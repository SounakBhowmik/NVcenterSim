from __future__ import annotations

"""Sweep utilities for sensitivity analyses."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class SweepStats:
    mean: float
    std: float


def aggregate(values: List[float]) -> SweepStats:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return SweepStats(mean=float("nan"), std=float("nan"))
    return SweepStats(
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    )


def summarize_metric_grid(grid: Dict[float, Dict[int, List[float]]]) -> Dict[float, Dict[int, SweepStats]]:
    """Aggregate {outer -> {inner -> [vals]}} into mean/std objects."""
    out: Dict[float, Dict[int, SweepStats]] = {}
    for outer_key, inner in grid.items():
        out[outer_key] = {inner_key: aggregate(vals) for inner_key, vals in inner.items()}
    return out
