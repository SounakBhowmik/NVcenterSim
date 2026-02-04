from __future__ import annotations
from typing import Dict
import numpy as np

_SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def bloch_from_density_matrix(rho: np.ndarray) -> np.ndarray:
    x = np.trace(rho @ _SIGMA_X).real
    y = np.trace(rho @ _SIGMA_Y).real
    z = np.trace(rho @ _SIGMA_Z).real
    return np.array([x, y, z], dtype=np.float64)


def p0_from_counts(counts: Dict[str, int], shots: int) -> float:
    return float(counts.get("0", 0)) / float(shots)


def exp_from_counts(counts: Dict[str, int], shots: int) -> float:
    """
    For 1-qubit measurement in Z basis, expectation of Z is:
      <Z> = p(0) - p(1) = 2*p(0) - 1
    For X or Y basis measurement circuits, same formula gives <X> or <Y>.
    """
    p0 = p0_from_counts(counts, shots)
    return 2.0 * p0 - 1.0
