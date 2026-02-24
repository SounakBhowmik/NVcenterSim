from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    b_min_t: float = 0.0
    b_max_t: float = 2e-6
    times_us: Tuple[float, ...] = (5, 10, 20, 40, 60)

    n_train: int = 800
    n_test: int = 400

    seeds: Tuple[int, ...] = (1, 2, 3)

    optimization_level: int = 2
    shots_list: Tuple[int, ...] = (64, 128, 256, 512, 1024)

    # NV parameters
    t1_s: float = 0.19392
    t2_s: float = 98.47e-6
    readout_err: float = 0.10

    # Baseline/model settings
    ridge_lambda: float = 1e-6
    kernel_lambda: float = 1e-6

    # New feature flags
    match_measurement_budget: bool = False
    enable_nonlinear_baselines: bool = True
    enable_tomo_kernel: bool = True
    enable_t2_sweep: bool = True

    # Sensitivity analysis settings
    t2_sweep_values: Tuple[float, ...] = (20e-6, 50e-6, 100e-6, 200e-6)
    t2_sweep_fixed_shots: Tuple[int, ...] = (128, 1024)
