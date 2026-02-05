from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _sqrtm_psd_2x2(A: np.ndarray) -> np.ndarray:
    """Matrix square root for 2x2 Hermitian PSD matrix."""
    # Eigen-decomposition is robust for 2x2
    w, v = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.conj().T


def fidelity_qubit(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Uhlmann fidelity for 2x2 density matrices.
    Returns F(rho, sigma) in [0,1].
    """
    # Ensure Hermitian numerically
    rho = 0.5 * (rho + rho.conj().T)
    sigma = 0.5 * (sigma + sigma.conj().T)

    srho = _sqrtm_psd_2x2(rho)
    inner = srho @ sigma @ srho
    sinner = _sqrtm_psd_2x2(inner)
    F = np.trace(sinner).real
    # Numerical safety
    return float(np.clip(F, 0.0, 1.0))


def kernel_between_samples(rhos_i: np.ndarray, rhos_j: np.ndarray) -> float:
    """
    rhos_i: [M,2,2], rhos_j: [M,2,2]
    k(i,j) = mean_k F(rho_i_k, rho_j_k)^2
    """
    M = rhos_i.shape[0]
    vals = []
    for k in range(M):
        F = fidelity_qubit(rhos_i[k], rhos_j[k])
        vals.append(F * F)
    return float(np.mean(vals))


def build_kernel_matrix(RA: np.ndarray, RB: np.ndarray | None = None) -> np.ndarray:
    """
    RA: [NA,M,2,2], RB: [NB,M,2,2] or None (then RB=RA)
    Returns K: [NA,NB]
    """
    if RB is None:
        RB = RA
    NA = RA.shape[0]
    NB = RB.shape[0]
    K = np.zeros((NA, NB), dtype=np.float64)
    for i in range(NA):
        for j in range(NB):
            K[i, j] = kernel_between_samples(RA[i], RB[j])
    return K


@dataclass
class KernelRidgeModel:
    alpha: np.ndarray  # [N_train]
    y_mean: float      # centering helps
    lam: float
    R_train: np.ndarray  # keep training states for test kernel


def fit_kernel_ridge(R_train: np.ndarray, y_train: np.ndarray, lam: float) -> KernelRidgeModel:
    """
    Kernel ridge regression:
      alpha = (K + lam I)^-1 (y - y_mean)
      y_hat(x) = y_mean + K(x, train) @ alpha
    """
    y_mean = float(np.mean(y_train))
    y0 = y_train - y_mean
    K = build_kernel_matrix(R_train)  # [N,N]
    A = K + lam * np.eye(K.shape[0])
    alpha = np.linalg.solve(A, y0)
    return KernelRidgeModel(alpha=alpha, y_mean=y_mean, lam=lam, R_train=R_train)


def predict_kernel_ridge(model: KernelRidgeModel, R_test: np.ndarray) -> np.ndarray:
    Kxt = build_kernel_matrix(R_test, model.R_train)  # [Ntest,Ntrain]
    return model.y_mean + Kxt @ model.alpha
