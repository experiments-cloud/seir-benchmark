"""
sparsity_benchmark_ekf.py

Data-sparsity benchmark for SEIR parameter recovery -- Extended Kalman
Filter (EKF) baseline.

Mirrors the experimental design of the companion NLS and EINN sparsity
benchmarks exactly: the same synthetic ground-truth outbreak, the same
five data-sparsity levels, and the same number of random-initialization
repeats per level. For the EKF, "sparsity" means the filter only receives
observations up to the given fraction of the full epidemic cycle; beyond
that point, no further updates are made and the final filtered parameter
estimate at the last available observation is reported (the EKF is an
online/sequential estimator, so there is no equivalent of "extrapolating"
the fit the way NLS or the EINN can -- its estimate reflects only what it
has filtered so far).

See ekf_seir.py for the filter implementation and the rationale behind
its default tuning (process noise levels).


Outputs are written to ./results/ (created automatically).

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seir_model import determine_outbreak_duration, solve_seir
from ekf_seir import run_ekf

# ---------------------------------------------------------------------------
# Configuration (kept identical to the companion NLS/EINN scripts wherever
# it affects comparability)
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

POPULATION_SIZE = 1_000_000
BETA_TRUE = 0.35
SIGMA_TRUE = 1 / 5.2
GAMMA_TRUE = 1 / 10.0
PARAM_NAMES = ["beta", "sigma", "gamma"]
PARAMS_TRUE = np.array([BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE])

E0, I0, R0 = 1, 1, 0
S0 = POPULATION_SIZE - E0 - I0 - R0
Y0 = np.array([S0, E0, I0, R0], dtype=float)

DATA_FRACTIONS = [0.15, 0.30, 0.45, 0.60, 1.00]
N_SEEDS = 15
OBSERVATION_INTERVAL_DAYS = 1.0

PARAM_BOUNDS_LOWER = np.array([0.01, 1 / 14, 1 / 21])
PARAM_BOUNDS_UPPER = np.array([2.00, 1 / 2, 1 / 3])

# Fixed, low observation noise (this experiment isolates the effect of
# sparsity; noise robustness is characterized separately).
OBSERVATION_NOISE_LEVEL = 0.01

RANDOM_SEED_BASE = 12345


# ---------------------------------------------------------------------------
# Synthetic ground truth
# ---------------------------------------------------------------------------

def generate_full_trajectory():
    full_duration = determine_outbreak_duration(PARAMS_TRUE, Y0, POPULATION_SIZE)
    n_points = int(np.ceil(full_duration / OBSERVATION_INTERVAL_DAYS)) + 1
    time_grid = np.linspace(0, full_duration, n_points)
    trajectory = solve_seir(PARAMS_TRUE, time_grid, Y0, POPULATION_SIZE)
    return time_grid, trajectory, full_duration


def sample_initial_guess(rng):
    log_lower = np.log(PARAM_BOUNDS_LOWER)
    log_upper = np.log(PARAM_BOUNDS_UPPER)
    return np.exp(rng.uniform(log_lower, log_upper))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def main():
    time_grid, full_trajectory, full_duration = generate_full_trajectory()
    infectious_curve = full_trajectory[2]

    print(f"Full outbreak duration: {full_duration:.1f} days "
          f"({len(time_grid)} daily observations).")

    raw_results = []

    for fraction in DATA_FRACTIONS:
        n_obs = max(int(np.floor(fraction * len(time_grid))), 5)
        i_obs = infectious_curve[:n_obs]
        obs_noise_std = np.maximum(OBSERVATION_NOISE_LEVEL * np.abs(i_obs), 1.0)

        for seed_idx in range(N_SEEDS):
            rng = np.random.default_rng(RANDOM_SEED_BASE + seed_idx)
            initial_guess = sample_initial_guess(rng)

            _, fitted_params = run_ekf(
                i_obs, dt=OBSERVATION_INTERVAL_DAYS, y0=Y0,
                param_init=initial_guess, population=POPULATION_SIZE,
                rho=1.0, observation_noise_std=obs_noise_std,
            )
            relative_error = np.abs(fitted_params - PARAMS_TRUE) / PARAMS_TRUE
            raw_results.append({
                "fraction": fraction,
                "seed": seed_idx,
                "beta_fitted": fitted_params[0],
                "sigma_fitted": fitted_params[1],
                "gamma_fitted": fitted_params[2],
                "beta_rel_error": relative_error[0],
                "sigma_rel_error": relative_error[1],
                "gamma_rel_error": relative_error[2],
                "mean_rel_error": float(np.mean(relative_error)),
            })

        print(f"Fraction {fraction:.0%}: {n_obs} observations, "
              f"{N_SEEDS} EKF fits completed.")

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "ekf_raw_results.csv"), index=False)

    summary_rows = []
    for fraction in DATA_FRACTIONS:
        subset = raw_df[raw_df["fraction"] == fraction]
        summary_rows.append({
            "fraction": fraction,
            "n_seeds": len(subset),
            "median_mean_rel_error": subset["mean_rel_error"].median(),
            "q25_mean_rel_error": subset["mean_rel_error"].quantile(0.25),
            "q75_mean_rel_error": subset["mean_rel_error"].quantile(0.75),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "ekf_summary.csv"), index=False)

    print("\nSummary (EKF):")
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["fraction"] * 100, summary_df["median_mean_rel_error"],
            marker="^", color="seagreen", label="Median relative error (EKF)")
    ax.fill_between(
        summary_df["fraction"] * 100,
        summary_df["q25_mean_rel_error"],
        summary_df["q75_mean_rel_error"],
        alpha=0.2, color="seagreen", label="Interquartile range",
    )
    ax.set_xlabel("Fraction of the epidemic curve used for fitting (%)")
    ax.set_ylabel("Mean relative parameter-recovery error")
    ax.set_title("SEIR parameter recovery under data sparsity -- EKF baseline")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "ekf_sparsity_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
