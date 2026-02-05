# run_nv_magnetometry.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from config import ExperimentConfig
from backend import build_nv_backend
from simulator import RamseySimulator
from dataset import generate_dataset
from ridge import fit_ridge, predict, mse
from qkernel import fit_kernel_ridge, predict_kernel_ridge


def rmse_uT(mse_val: float) -> float:
    return 1e6 * math.sqrt(mse_val)


@dataclass
class AggStats:
    mean: float
    std: float


def agg(x: List[float]) -> AggStats:
    arr = np.array(x, dtype=np.float64)
    return AggStats(mean=float(np.mean(arr)), std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)


def main():
    cfg = ExperimentConfig(
        # Keep these as you like
        shots_list=(8, 16, 64, 128, 256, 512, 1024, 2048),
        seeds=(11, 45, 33),  # 3-seed sweep
        n_train=2000,
        n_test=200,
        kernel_lambda=1e-4,  # if kernel curve wiggles, try 1e-5 or 1e-4
        readout_err=0.10,
        times_us=(5, 10, 20, 40, 60),
        b_max_t=2e-6,
    )

    # Build NV backend once (noise params fixed) and reuse across seeds
    nv_backend = build_nv_backend(cfg.t1_s, cfg.t2_s, cfg.readout_err)
    sim = RamseySimulator(nv_backend, optimization_level=cfg.optimization_level)

    # Collect per-shot per-seed MSEs
    per_shot_mse: Dict[int, Dict[str, List[float]]] = {}
    for shots in cfg.shots_list:
        per_shot_mse[shots] = {"cz": [], "cxyz": [], "k": []}

    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)

        for shots in cfg.shots_list:
            print(f"Running seed={seed}, shots={shots} ...")

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

            # Classical(Z)
            m_cz = fit_ridge(train.X_classical_z, train.y_B, lam=cfg.ridge_lambda)
            p_cz = predict(m_cz, test.X_classical_z)
            mse_cz = mse(test.y_B, p_cz)

            # Classical(XYZ)
            m_cxyz = fit_ridge(train.X_classical_xyz, train.y_B, lam=cfg.ridge_lambda)
            p_cxyz = predict(m_cxyz, test.X_classical_xyz)
            mse_cxyz = mse(test.y_B, p_cxyz)

            # Quantum kernel (fidelity-kernel ridge)
            m_k = fit_kernel_ridge(train.R_quantum, train.y_B, lam=cfg.kernel_lambda)
            p_k = predict_kernel_ridge(m_k, test.R_quantum)
            mse_k = mse(test.y_B, p_k)

            per_shot_mse[shots]["cz"].append(mse_cz)
            per_shot_mse[shots]["cxyz"].append(mse_cxyz)
            per_shot_mse[shots]["k"].append(mse_k)

    # Aggregate and print summary
    print("\n=== Summary (mean ± std over seeds) ===")
    shots_sorted = list(cfg.shots_list)

    stats = {
        "cz": {"mse": [], "rmse": []},
        "cxyz": {"mse": [], "rmse": []},
        "k": {"mse": [], "rmse": []},
    }

    for shots in shots_sorted:
        s_cz = agg(per_shot_mse[shots]["cz"])
        s_cxyz = agg(per_shot_mse[shots]["cxyz"])
        s_k = agg(per_shot_mse[shots]["k"])

        # store for plotting (RMSE)
        stats["cz"]["mse"].append((s_cz.mean, s_cz.std))
        stats["cxyz"]["mse"].append((s_cxyz.mean, s_cxyz.std))
        stats["k"]["mse"].append((s_k.mean, s_k.std))

        cz_rmse_mean = rmse_uT(s_cz.mean)
        cxyz_rmse_mean = rmse_uT(s_cxyz.mean)
        k_rmse_mean = rmse_uT(s_k.mean)

        # approx std propagation for RMSE: std_rmse ≈ (1/(2*sqrt(mse))) * std_mse
        def rmse_std(mse_mean: float, mse_std: float) -> float:
            if mse_mean <= 0:
                return 0.0
            return 1e6 * (0.5 / math.sqrt(mse_mean)) * mse_std

        cz_rmse_std = rmse_std(s_cz.mean, s_cz.std)
        cxyz_rmse_std = rmse_std(s_cxyz.mean, s_cxyz.std)
        k_rmse_std = rmse_std(s_k.mean, s_k.std)

        stats["cz"]["rmse"].append((cz_rmse_mean, cz_rmse_std))
        stats["cxyz"]["rmse"].append((cxyz_rmse_mean, cxyz_rmse_std))
        stats["k"]["rmse"].append((k_rmse_mean, k_rmse_std))

        print(
            f"shots={shots:4d} | "
            f"mse_cz={s_cz.mean:.3e}±{s_cz.std:.1e} | "
            f"mse_cxyz={s_cxyz.mean:.3e}±{s_cxyz.std:.1e} | "
            f"mse_k={s_k.mean:.3e}±{s_k.std:.1e} | "
            f"RMSE(µT): cz={cz_rmse_mean:.4f}±{cz_rmse_std:.4f}, "
            f"cxyz={cxyz_rmse_mean:.4f}±{cxyz_rmse_std:.4f}, "
            f"k={k_rmse_mean:.4f}±{k_rmse_std:.4f}"
        )

    # Plot RMSE vs shots
    x = np.array(shots_sorted, dtype=np.float64)

    def plot_series(label: str, key: str):
        y = np.array([v[0] for v in stats[key]["rmse"]], dtype=np.float64)
        yerr = np.array([v[1] for v in stats[key]["rmse"]], dtype=np.float64)
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)

    plt.figure()
    plot_series("ML-C (Z only)", "cz")
    plot_series("ML-C (XYZ enriched)", "cxyz")
    plot_series("QML-Q (fidelity kernel)", "k")

    plt.xscale("log", base=2)
    plt.yscale("log")  # log y makes differences clear; remove if you prefer linear
    plt.xlabel("Shots")
    plt.ylabel("Test RMSE on B (µT)")
    plt.title("NV-inspired magnetometry regression (mean ± std over seeds)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    out = "rmse_vs_shots.png"
    plt.savefig(out, dpi=220)
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
