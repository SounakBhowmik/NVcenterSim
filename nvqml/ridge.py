from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RidgeModel:
    w: np.ndarray  # [D]
    b: float


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> RidgeModel:
    """
    Ridge regression with bias term (bias not regularized).
      minimize ||Xw + b - y||^2 + lam||w||^2
    """
    N, D = X.shape
    X_aug = np.hstack([X, np.ones((N, 1), dtype=X.dtype)])
    reg = np.diag(np.concatenate([np.full(D, lam), np.array([0.0])]))
    A = X_aug.T @ X_aug + reg
    rhs = X_aug.T @ y
    theta = np.linalg.solve(A, rhs)
    return RidgeModel(w=theta[:D], b=float(theta[D]))


def predict(model: RidgeModel, X: np.ndarray) -> np.ndarray:
    return X @ model.w + model.b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_true - y_pred
    return float(np.mean(d * d))
