from __future__ import annotations
from typing import Dict
import numpy as np

_SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def bloch_from_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Return Bloch vector [x, y, z] = [Tr(rho X), Tr(rho Y), Tr(rho Z)]."""
    x = np.trace(rho @ _SIGMA_X).real
    y = np.trace(rho @ _SIGMA_Y).real
    z = np.trace(rho @ _SIGMA_Z).real
    return np.array([x, y, z], dtype=np.float64)


def p0_from_counts(counts: Dict[str, int], shots: int) -> float:
    """Estimate p0 from Qiskit counts dict."""
    return float(counts.get("0", 0)) / float(shots)
