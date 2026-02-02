from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .circuits import phase_from_B, build_ramsey_circuit
from .features import bloch_from_density_matrix, p0_from_counts
from .simulator import RamseySimulator


@dataclass
class Dataset:
    X_classical: np.ndarray  # [N, M]
    X_quantum: np.ndarray    # [N, 3M]
    y_B: np.ndarray          # [N]


def generate_dataset(
    sim: RamseySimulator,
    rng: np.random.Generator,
    n_samples: int,
    shots: int,
    b_min_t: float,
    b_max_t: float,
    times_us: Tuple[float, ...],
) -> Dataset:
    M = len(times_us)
    Xc = np.zeros((n_samples, M), dtype=np.float64)
    Xq = np.zeros((n_samples, 3 * M), dtype=np.float64)
    y = np.zeros((n_samples,), dtype=np.float64)

    times_s = [t * 1e-6 for t in times_us]

    for i in range(n_samples):
        B = float(rng.uniform(b_min_t, b_max_t))
        y[i] = B

        for k, t_s in enumerate(times_s):
            phi = phase_from_B(B, t_s)
            qc = build_ramsey_circuit(phi)

            out = sim.run(qc, shots=shots, dm_label="rho")

            Xc[i, k] = p0_from_counts(out.counts, shots)
            Xq[i, 3 * k : 3 * k + 3] = bloch_from_density_matrix(out.density_matrix)

    return Dataset(X_classical=Xc, X_quantum=Xq, y_B=y)
