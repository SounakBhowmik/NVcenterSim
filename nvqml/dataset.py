# nvqml/dataset.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# from .circuits import phase_from_B, build_ramsey_circuit
# from .features import bloch_from_density_matrix, p0_from_counts, exp_from_counts
# from .simulator import RamseySimulator

from circuits import phase_from_B, build_ramsey_circuit
from features import bloch_from_density_matrix, p0_from_counts, exp_from_counts
from simulator import RamseySimulator

@dataclass
class Dataset:
    X_classical_z: np.ndarray    # [N, M] using p0(tk)
    X_classical_xyz: np.ndarray  # [N, 3M] using <X>,<Y>,<Z> at each tk
    X_quantum: np.ndarray        # [N, 3M] from density matrix Bloch
    y_B: np.ndarray              # [N]


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
    # X_classical_z = np.zeros((n_samples, M), dtype=np.float64)
    # X_classical_xyz = np.zeros((n_samples, 3 * M), dtype=np.float64)
    # X_quantum = np.zeros((n_samples, 3 * M), dtype=np.float64)

    Xz = np.zeros((n_samples, M), dtype=np.float64)
    Xxyz = np.zeros((n_samples, 3 * M), dtype=np.float64)
    Xq = np.zeros((n_samples, 3 * M), dtype=np.float64)
    y = np.zeros((n_samples,), dtype=np.float64)

    times_s = [t * 1e-6 for t in times_us]

    for i in range(n_samples):
        B = float(rng.uniform(b_min_t, b_max_t))
        y[i] = B

        for k, t_s in enumerate(times_s):
            phi = phase_from_B(B, t_s)

            # --- Classical Z-only ---
            qc_z = build_ramsey_circuit(phi, meas_basis="Z")
            out_z = sim.run(qc_z, shots=shots, dm_label="rho")
            Xz[i, k] = p0_from_counts(out_z.counts, shots)

            # --- Quantum Bloch from rho (same run as Z-only circuit) ---
            # Note: rho here corresponds to state right before measurement in the Z-variant circuit.
            Xq[i, 3 * k : 3 * k + 3] = bloch_from_density_matrix(out_z.density_matrix)

            # --- Classical XYZ enriched measurements ---
            # We run extra circuits for X and Y measurements (still classical data).
            qc_x = build_ramsey_circuit(phi, meas_basis="X")
            out_x = sim.run(qc_x, shots=shots, dm_label="rho")  # rho unused here
            ex = exp_from_counts(out_x.counts, shots)

            qc_y = build_ramsey_circuit(phi, meas_basis="Y")
            out_y = sim.run(qc_y, shots=shots, dm_label="rho")  # rho unused here
            ey = exp_from_counts(out_y.counts, shots)

            # For Z expectation, use the Z run:
            ez = exp_from_counts(out_z.counts, shots)

            Xxyz[i, 3 * k : 3 * k + 3] = np.array([ex, ey, ez], dtype=np.float64)

    return Dataset(X_classical_z=Xz, X_classical_xyz=Xxyz, X_quantum=Xq, y_B=y)
