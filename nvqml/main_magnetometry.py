from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from backend import build_nv_backend
from config import ExperimentConfig
from dataset import generate_dataset
from models_classical import fit_mlp, fit_rbf_krr, predict_model
from qkernel import fit_kernel_ridge, predict_kernel_ridge
from ridge import fit_ridge, mse, predict
from simulator import RamseySimulator
from sweeps import summarize_metric_grid


def rmse_uT(mse_val: float) -> float:
    return 1e6 * math.sqrt(max(mse_val, 0.0))


@dataclass
class AggStats:
    mean: float
    std: float


def agg(x: List[float]) -> AggStats:
    arr = np.array(x, dtype=np.float64)
    return AggStats(mean=float(np.mean(arr)), std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)


def run_core_experiment(cfg: ExperimentConfig, t2_override: float | None = None) -> Dict[int, Dict[str, List[float]]]:
    t2_s = cfg.t2_s if t2_override is None else t2_override
    nv_backend = build_nv_backend(cfg.t1_s, t2_s, cfg.readout_err)
    sim = RamseySimulator(nv_backend, optimization_level=cfg.optimization_level)

    keys = [
        "lin_z",
        "lin_xyz",
        "rbf_z",
        "mlp_z",
        "best_nonlin_z",
        "rbf_xyz",
        "mlp_xyz",
        "best_nonlin_xyz",
        "rbf_bloch",
        "mlp_bloch",
        "best_nonlin_bloch",
        "k_oracle",
        "k_tomo",
    ]
    per_shot_mse: Dict[int, Dict[str, List[float]]] = {shots: {k: [] for k in keys} for shots in cfg.shots_list}

    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)

        for shots in cfg.shots_list:
            print(f"Running seed={seed}, shots={shots}, T2={t2_s:.2e} ...")
            train = generate_dataset(
                sim=sim,
                rng=rng,
                n_samples=cfg.n_train,
                shots=shots,
                b_min_t=cfg.b_min_t,
                b_max_t=cfg.b_max_t,
                times_us=cfg.times_us,
                match_measurement_budget=cfg.match_measurement_budget,
            )
            test = generate_dataset(
                sim=sim,
                rng=rng,
                n_samples=cfg.n_test,
                shots=shots,
                b_min_t=cfg.b_min_t,
                b_max_t=cfg.b_max_t,
                times_us=cfg.times_us,
                match_measurement_budget=cfg.match_measurement_budget,
            )

            m_lin_z = fit_ridge(train.X_classical_z, train.y_B, lam=cfg.ridge_lambda)
            mse_lin_z = mse(test.y_B, predict(m_lin_z, test.X_classical_z))

            m_lin_xyz = fit_ridge(train.X_classical_xyz, train.y_B, lam=cfg.ridge_lambda)
            mse_lin_xyz = mse(test.y_B, predict(m_lin_xyz, test.X_classical_xyz))

            per_shot_mse[shots]["lin_z"].append(mse_lin_z)
            per_shot_mse[shots]["lin_xyz"].append(mse_lin_xyz)

            if cfg.enable_nonlinear_baselines:
                m_rbf_z = fit_rbf_krr(train.X_classical_z, train.y_B, alpha=cfg.ridge_lambda)
                mse_rbf_z = mse(test.y_B, predict_model(m_rbf_z, test.X_classical_z))
                m_mlp_z = fit_mlp(train.X_classical_z, train.y_B, seed=seed)
                mse_mlp_z = mse(test.y_B, predict_model(m_mlp_z, test.X_classical_z))

                m_rbf_xyz = fit_rbf_krr(train.X_classical_xyz, train.y_B, alpha=cfg.ridge_lambda)
                mse_rbf_xyz = mse(test.y_B, predict_model(m_rbf_xyz, test.X_classical_xyz))
                m_mlp_xyz = fit_mlp(train.X_classical_xyz, train.y_B, seed=seed)
                mse_mlp_xyz = mse(test.y_B, predict_model(m_mlp_xyz, test.X_classical_xyz))

                m_rbf_bloch = fit_rbf_krr(train.X_bloch, train.y_B, alpha=cfg.ridge_lambda)
                mse_rbf_bloch = mse(test.y_B, predict_model(m_rbf_bloch, test.X_bloch))
                m_mlp_bloch = fit_mlp(train.X_bloch, train.y_B, seed=seed)
                mse_mlp_bloch = mse(test.y_B, predict_model(m_mlp_bloch, test.X_bloch))

                per_shot_mse[shots]["rbf_z"].append(mse_rbf_z)
                per_shot_mse[shots]["mlp_z"].append(mse_mlp_z)
                per_shot_mse[shots]["best_nonlin_z"].append(min(mse_rbf_z, mse_mlp_z))

                per_shot_mse[shots]["rbf_xyz"].append(mse_rbf_xyz)
                per_shot_mse[shots]["mlp_xyz"].append(mse_mlp_xyz)
                per_shot_mse[shots]["best_nonlin_xyz"].append(min(mse_rbf_xyz, mse_mlp_xyz))

                per_shot_mse[shots]["rbf_bloch"].append(mse_rbf_bloch)
                per_shot_mse[shots]["mlp_bloch"].append(mse_mlp_bloch)
                per_shot_mse[shots]["best_nonlin_bloch"].append(min(mse_rbf_bloch, mse_mlp_bloch))

            m_k_oracle = fit_kernel_ridge(train.R_oracle, train.y_B, lam=cfg.kernel_lambda)
            mse_k_oracle = mse(test.y_B, predict_kernel_ridge(m_k_oracle, test.R_oracle))
            per_shot_mse[shots]["k_oracle"].append(mse_k_oracle)

            if cfg.enable_tomo_kernel:
                m_k_tomo = fit_kernel_ridge(train.R_tomo, train.y_B, lam=cfg.kernel_lambda)
                mse_k_tomo = mse(test.y_B, predict_kernel_ridge(m_k_tomo, test.R_tomo))
                per_shot_mse[shots]["k_tomo"].append(mse_k_tomo)

    return per_shot_mse


def print_and_plot_summary(cfg: ExperimentConfig, per_shot_mse: Dict[int, Dict[str, List[float]]]) -> None:
    print("\n=== Summary (mean ± std over seeds) ===")
    shots_sorted = list(cfg.shots_list)
    methods = [
        ("lin_z", "Linear Ridge (Z)"),
        ("lin_xyz", "Linear Ridge (XYZ)"),
        ("best_nonlin_z", "Best nonlinear (Z)"),
        ("best_nonlin_xyz", "Best nonlinear (XYZ)"),
        ("best_nonlin_bloch", "Best nonlinear (Bloch)"),
        ("k_oracle", "KRR (oracle ρ)"),
        ("k_tomo", "KRR (tomo ρ̂)"),
    ]

    stats_rmse: Dict[str, List[tuple[float, float]]] = {k: [] for k, _ in methods}

    for shots in shots_sorted:
        parts = [f"shots={shots:4d}"]
        for key, label in methods:
            vals = per_shot_mse[shots][key]
            if not vals:
                stats_rmse[key].append((math.nan, math.nan))
                continue
            s = agg(vals)
            rmse_mean = rmse_uT(s.mean)
            rmse_std = 0.0 if s.mean <= 0 else 1e6 * (0.5 / math.sqrt(s.mean)) * s.std
            stats_rmse[key].append((rmse_mean, rmse_std))
            parts.append(f"{label}: mse={s.mean:.3e}±{s.std:.1e}, rmse={rmse_mean:.4f}±{rmse_std:.4f} µT")
        print(" | ".join(parts))

    plt.figure(figsize=(9.5, 6.5))
    x = np.array(shots_sorted, dtype=np.float64)

    curve_specs = [
        ("lin_z", "Linear Ridge (Z only)"),
        ("lin_xyz", "Linear Ridge (XYZ)"),
        ("rbf_xyz", "RBF-KRR (XYZ)"),
        ("mlp_xyz", "MLP (XYZ)"),
        ("best_nonlin_bloch", "RBF/MLP (Bloch)"),
        ("k_oracle", "KRR (oracle ρ)"),
        ("k_tomo", "KRR (tomo ρ̂)"),
    ]

    for key, label in curve_specs:
        y = []
        yerr = []
        for shots in shots_sorted:
            vals = per_shot_mse[shots][key]
            if not vals:
                y.append(np.nan)
                yerr.append(np.nan)
            else:
                s = agg(vals)
                y.append(rmse_uT(s.mean))
                yerr.append(0.0 if s.mean <= 0 else 1e6 * (0.5 / math.sqrt(s.mean)) * s.std)
        plt.errorbar(x, np.array(y), yerr=np.array(yerr), marker="o", capsize=3, label=label)

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Shots")
    plt.ylabel("Test RMSE on B (µT)")
    plt.title("NV-inspired magnetometry regression (mean ± std over seeds)")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = "rmse_vs_shots.png"
    plt.savefig(out, dpi=220)
    print(f"\nSaved plot: {out}")


def run_t2_sweep(cfg: ExperimentConfig) -> None:
    fixed_shots = cfg.t2_sweep_fixed_shots
    raw: Dict[float, Dict[int, List[float]]] = {
        t2: {shot: [] for shot in fixed_shots} for t2 in cfg.t2_sweep_values
    }

    for t2 in cfg.t2_sweep_values:
        sweep_cfg = ExperimentConfig(**{**cfg.__dict__, "shots_list": fixed_shots, "enable_t2_sweep": False})
        per_shot = run_core_experiment(sweep_cfg, t2_override=t2)
        for shot in fixed_shots:
            raw[t2][shot] = per_shot[shot]["k_tomo"] if per_shot[shot]["k_tomo"] else per_shot[shot]["k_oracle"]

    summary = summarize_metric_grid(raw)

    plt.figure(figsize=(8.0, 5.5))
    t2_us = np.array([1e6 * v for v in cfg.t2_sweep_values], dtype=np.float64)
    for shot in fixed_shots:
        means = np.array([rmse_uT(summary[t2][shot].mean) for t2 in cfg.t2_sweep_values])
        stds = np.array([
            0.0
            if summary[t2][shot].mean <= 0
            else 1e6 * (0.5 / math.sqrt(summary[t2][shot].mean)) * summary[t2][shot].std
            for t2 in cfg.t2_sweep_values
        ])
        plt.errorbar(t2_us, means, yerr=stds, marker="o", capsize=3, label=f"KRR (tomo ρ̂), shots={shot}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("T2 (µs)")
    plt.ylabel("RMSE (µT)")
    plt.title("T2 sensitivity: RMSE vs T2 at fixed shots")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    out = "rmse_vs_t2_fixed_shots.png"
    plt.savefig(out, dpi=220)
    print(f"Saved T2 sweep plot: {out}")


def main() -> None:
    cfg = ExperimentConfig(
        shots_list=(8, 16, 64, 128, 256, 512, 1024, 2048),
        seeds=(11, 45, 33),
        n_train=300,
        n_test=120,
        kernel_lambda=1e-4,
        readout_err=0.10,
        times_us=(5, 10, 20, 40, 60),
        b_max_t=2e-6,
        match_measurement_budget=True,
    )

    per_shot_mse = run_core_experiment(cfg)
    print_and_plot_summary(cfg, per_shot_mse)

    if cfg.enable_t2_sweep:
        run_t2_sweep(cfg)


if __name__ == "__main__":
    main()
