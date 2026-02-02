from __future__ import annotations
import numpy as np

from nvqml.config import ExperimentConfig
from nvqml.backend import build_nv_backend
from nvqml.simulator import RamseySimulator
from nvqml.dataset import generate_dataset
from nvqml.ridge import fit_ridge, predict, mse


def main():
    cfg = ExperimentConfig(
        # Start small for debugging; scale later
        n_train=200,
        n_test=80,
        shots_list=(64, 128),
        readout_err=0.10,
        times_us=(5, 10, 20, 40, 60),
    )

    rng = np.random.default_rng(cfg.seed)

    nv_backend = build_nv_backend(cfg.t1_s, cfg.t2_s, cfg.readout_err)
    sim = RamseySimulator(nv_backend, optimization_level=cfg.optimization_level)

    results = []
    for shots in cfg.shots_list:
        print(f"\n=== shots={shots} ===")

        train = generate_dataset(
            sim=sim, rng=rng,
            n_samples=cfg.n_train, shots=shots,
            b_min_t=cfg.b_min_t, b_max_t=cfg.b_max_t,
            times_us=cfg.times_us
        )
        test = generate_dataset(
            sim=sim, rng=rng,
            n_samples=cfg.n_test, shots=shots,
            b_min_t=cfg.b_min_t, b_max_t=cfg.b_max_t,
            times_us=cfg.times_us
        )

        # ML-C
        m_c = fit_ridge(train.X_classical, train.y_B, lam=cfg.ridge_lambda)
        p_c = predict(m_c, test.X_classical)
        mse_c = mse(test.y_B, p_c)

        # QML-Q (Bloch features)
        m_q = fit_ridge(train.X_quantum, train.y_B, lam=cfg.ridge_lambda)
        p_q = predict(m_q, test.X_quantum)
        mse_q = mse(test.y_B, p_q)

        print(f"MSE classical: {mse_c:.6e}")
        print(f"MSE quantum:   {mse_q:.6e}")
        results.append((shots, mse_c, mse_q))

    print("\n=== Summary ===")
    for shots, mc, mq in results:
        print(f"shots={shots:4d} | mse_c={mc:.3e} | mse_q={mq:.3e}")


if __name__ == "__main__":
    main()
