from __future__ import annotations

"""Utilities for tomography-limited single-qubit state reconstruction."""

import numpy as np

_SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_ID2 = np.eye(2, dtype=np.complex128)


def reconstruct_density_matrix(ex: float, ey: float, ez: float) -> np.ndarray:
    """Build a physical single-qubit density matrix from Bloch expectations.

    The initial matrix is defined as
      rho = 0.5 * (I + ex*X + ey*Y + ez*Z)
    and then projected to the nearest PSD Hermitian trace-1 matrix by clipping tiny
    negative eigenvalues and renormalizing.
    """
    rho = 0.5 * (_ID2 + ex * _SIGMA_X + ey * _SIGMA_Y + ez * _SIGMA_Z)
    rho = 0.5 * (rho + rho.conj().T)

    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals.real, 0.0, None)
    total = float(np.sum(eigvals))
    if total <= 0.0:
        return 0.5 * _ID2
    eigvals /= total
    return (eigvecs * eigvals) @ eigvecs.conj().T
