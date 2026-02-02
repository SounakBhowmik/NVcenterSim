from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    # Parameter range (Tesla)
    b_min_t: float = 0.0
    b_max_t: float = 2e-6  # 2 µT

    # Evolution times (microseconds)
    times_us: Tuple[float, ...] = (5, 10, 20, 40, 60)

    # Dataset sizes
    n_train: int = 500
    n_test: int = 200
    seed: int = 123

    # NV-like single-qubit parameters (deterministic for reproducibility)
    t1_s: float = 0.19392
    t2_s: float = 98.47e-6
    readout_err: float = 0.10

    # Simulation / transpile
    optimization_level: int = 2
    shots_list: Tuple[int, ...] = (32, 64, 128)

    # Ridge regression
    ridge_lambda: float = 1e-6
