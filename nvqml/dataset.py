from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from circuits import phase_from_B, build_ramsey_circuit
from features import p0_from_counts, exp_from_counts
from simulator import RamseySimulator
from tomography import reconstruct_density_matrix


@dataclass
class Dataset:
    X_classical_z: np.ndarray    # [N, M] using p0(tk)
    X_classical_xyz: np.ndarray  # [N, 3M] using <X>,<Y>,<Z> at each tk
    X_bloch: np.ndarray          # [N, 3M] Bloch vectors from finite-shot XYZ
    R_oracle: np.ndarray         # [N, M, 2, 2]
    R_tomo: np.ndarray           # [N, M, 2, 2]
    y_B: np.ndarray              # [N]


@dataclass(frozen=True)
class ShotAllocation:
    shots_z_only: int
    shots_x: int
    shots_y: int
    shots_z_xyz: int


def allocate_shots(total_shots: int, match_budget: bool) -> ShotAllocation:
    """Allocate per-basis shots for each (sample, time) point."""
    if not match_budget:
        return ShotAllocation(
            shots_z_only=total_shots,
            shots_x=total_shots,
            shots_y=total_shots,
            shots_z_xyz=total_shots,
        )

    base = total_shots // 3
    rem = total_shots - 3 * base
    sx = base + (1 if rem > 0 else 0)
    sy = base + (1 if rem > 1 else 0)
    sz = base
    return ShotAllocation(shots_z_only=total_shots, shots_x=sx, shots_y=sy, shots_z_xyz=sz)


def generate_dataset(
    sim: RamseySimulator,
    rng: np.random.Generator,
    n_samples: int,
    shots: int,
    b_min_t: float,
    b_max_t: float,
    times_us: Tuple[float, ...],
    match_measurement_budget: bool = False,
) -> Dataset:
    M = len(times_us)
    Xz = np.zeros((n_samples, M), dtype=np.float64)
    Xxyz = np.zeros((n_samples, 3 * M), dtype=np.float64)
    Xbloch = np.zeros((n_samples, 3 * M), dtype=np.float64)
    R_oracle = np.zeros((n_samples, M, 2, 2), dtype=np.complex128)
    R_tomo = np.zeros((n_samples, M, 2, 2), dtype=np.complex128)
    y = np.zeros((n_samples,), dtype=np.float64)

    times_s = [t * 1e-6 for t in times_us]
    alloc = allocate_shots(shots, match_measurement_budget)

    for i in range(n_samples):
        B = float(rng.uniform(b_min_t, b_max_t))
        y[i] = B

        for k, t_s in enumerate(times_s):
            phi = phase_from_B(B, t_s)

            qc_z = build_ramsey_circuit(phi, meas_basis="Z")
            out_z_oracle = sim.run(qc_z, shots=max(1, alloc.shots_z_only), dm_label="rho")
            Xz[i, k] = p0_from_counts(out_z_oracle.counts, max(1, alloc.shots_z_only))

            rho_c = np.asarray(out_z_oracle.density_matrix, dtype=np.complex128)
            rho_c = 0.5 * (rho_c + rho_c.conj().T)
            R_oracle[i, k, :, :] = rho_c

            # Tomography-limited XYZ estimates with split budget.
            out_z_xyz = sim.run(qc_z, shots=max(1, alloc.shots_z_xyz), dm_label="rho")
            qc_x = build_ramsey_circuit(phi, meas_basis="X")
            out_x = sim.run(qc_x, shots=max(1, alloc.shots_x), dm_label="rho")
            qc_y = build_ramsey_circuit(phi, meas_basis="Y")
            out_y = sim.run(qc_y, shots=max(1, alloc.shots_y), dm_label="rho")

            ex = exp_from_counts(out_x.counts, max(1, alloc.shots_x))
            ey = exp_from_counts(out_y.counts, max(1, alloc.shots_y))
            ez = exp_from_counts(out_z_xyz.counts, max(1, alloc.shots_z_xyz))

            xyz = np.array([ex, ey, ez], dtype=np.float64)
            Xxyz[i, 3 * k : 3 * k + 3] = xyz
            Xbloch[i, 3 * k : 3 * k + 3] = xyz
            R_tomo[i, k, :, :] = reconstruct_density_matrix(ex, ey, ez)

    return Dataset(
        X_classical_z=Xz,
        X_classical_xyz=Xxyz,
        X_bloch=Xbloch,
        R_oracle=R_oracle,
        R_tomo=R_tomo,
        y_B=y,
    )
