from __future__ import annotations
import numpy as np

from config import ExperimentConfig
from backend import build_nv_backend
from simulator import RamseySimulator
from dataset import generate_dataset
from ridge import fit_ridge, predict, mse
import math


def rmse_uT(mse_val: float) -> float:
    return 1e6 * math.sqrt(mse_val)

def main():
    cfg = ExperimentConfig(
        # Start small for debugging; scale later
        n_train=200,
        n_test=80,
        shots_list=(64, 128, 256, 512, 1024, 2048),
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

        # ML-C (Z)
        m_cz = fit_ridge(train.X_classical_z, train.y_B, lam=cfg.ridge_lambda)
        p_cz = predict(m_cz, test.X_classical_z)
        mse_cz = mse(test.y_B, p_cz)

        # ML-C (XYZ enriched)
        m_cxyz = fit_ridge(train.X_classical_xyz, train.y_B, lam=cfg.ridge_lambda)
        p_cxyz = predict(m_cxyz, test.X_classical_xyz)
        mse_cxyz = mse(test.y_B, p_cxyz)

        # QML-Q (Bloch from rho)
        m_q = fit_ridge(train.X_quantum, train.y_B, lam=cfg.ridge_lambda)
        p_q = predict(m_q, test.X_quantum)
        mse_q = mse(test.y_B, p_q)

        print(f"MSE classical(Z):    {mse_cz:.6e} | RMSE: {rmse_uT(mse_cz):.4f} µT")
        print(f"MSE classical(XYZ):  {mse_cxyz:.6e} | RMSE: {rmse_uT(mse_cxyz):.4f} µT")
        print(f"MSE quantum(Bloch):  {mse_q:.6e} | RMSE: {rmse_uT(mse_q):.4f} µT")

        # print(f"MSE classical: {mse_c:.6e}")
        # print(f"MSE quantum:   {mse_q:.6e}")
        results.append((shots, mse_cz, mse_cxyz, mse_q))

    print("\n=== Summary ===")
    for shots, mse_cz, mse_cxyz, mse_q in results:
        print(f"shots={shots:4d} | mse_cz={mse_cz:.3e} | mse_cxyz={mse_cxyz:.3e} | mse_q={mse_q:.3e}")


if __name__ == "__main__":
    main()
