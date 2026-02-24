from __future__ import annotations

"""Classical baseline models used in the NV-QML regression experiments."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ModelName = Literal["rbf_krr", "mlp"]


@dataclass
class SKLearnModel:
    model: Pipeline


def fit_rbf_krr(X: np.ndarray, y: np.ndarray, alpha: float, gamma: float | None = None) -> SKLearnModel:
    """Fit RBF-kernel ridge with standardization."""
    if gamma is None:
        gamma = 1.0 / max(X.shape[1], 1)
    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("krr", KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)),
        ]
    )
    pipe.fit(X, y)
    return SKLearnModel(model=pipe)


def fit_mlp(X: np.ndarray, y: np.ndarray, seed: int) -> SKLearnModel:
    """Fit a compact MLP regressor with standardization."""
    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=600,
                    random_state=seed,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return SKLearnModel(model=pipe)


def predict_model(model: SKLearnModel, X: np.ndarray) -> np.ndarray:
    return model.model.predict(X)
