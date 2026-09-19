"""
noise_benchmark_ekf.py

Observation-noise robustness benchmark for SEIR parameter recovery --
Extended Kalman Filter (EKF) baseline.

Mirrors the experimental design of the companion NLS and EINN noise
benchmarks: the same synthetic ground-truth outbreak, observed over the
full epidemic cycle, the same noise levels, and the same number of random
noise-realization / initialization repeats per level. Unlike the NLS and
EINN scripts, the EKF is given its TRUE per-step observation noise
standard deviation (the same one used to generate the noise), since the
EKF is a filtering algorithm that requires an explicit noise model as
input, rather than something it infers from the data the way a
least-squares or physics-informed fit does implicitly. This is noted here
because it means the EKF has access to information the other two methods
do not (the true noise level) -- an asymmetry worth flagging explicitly
in the manuscript rather than leaving implicit, since it could otherwise
look like an unfair advantage or be missed by a reviewer.


Outputs are written to ./results/ (created automatically).

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seir_model import determine_outbreak_duration, solve_seir
from ekf_seir import run_ekf

# ---------------------------------------------------------------------------
# Configuration
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

NOISE_LEVELS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 1.00]
NOISE_FLOOR_CASES = 1.0

N_SEEDS = 15
OBSERVATION_INTERVAL_DAYS = 1.0

PARAM_BOUNDS_LOWER = np.array([0.01, 1 / 14, 1 / 21])
PARAM_BOUNDS_UPPER = np.array([2.00, 1 / 2, 1 / 3])

RANDOM_SEED_BASE = 12345


# ---------------------------------------------------------------------------
# Synthetic ground truth and noise model (identical to the companion NLS
# and EINN noise-benchmark scripts)
# ---------------------------------------------------------------------------

def generate_full_trajectory():
    full_duration = determine_outbreak_duration(PARAMS_TRUE, Y0, POPULATION_SIZE)
    n_points = int(np.ceil(full_duration / OBSERVATION_INTERVAL_DAYS)) + 1
    time_grid = np.linspace(0, full_duration, n_points)
    trajectory = solve_seir(PARAMS_TRUE, time_grid, Y0, POPULATION_SIZE)
    return time_grid, trajectory, full_duration


def add_noise(i_true, noise_level, rng):
    if noise_level == 0.0:
        return i_true.copy(), np.full_like(i_true, NOISE_FLOOR_CASES)
    noise_std = np.maximum(noise_level * np.abs(i_true), NOISE_FLOOR_CASES)
    noisy = i_true + rng.normal(0.0, noise_std)
    return np.clip(noisy, 0.0, None), noise_std


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

    for noise_level in NOISE_LEVELS:
        for seed_idx in range(N_SEEDS):
            rng = np.random.default_rng(
                RANDOM_SEED_BASE + seed_idx + int(noise_level * 10000)
            )
            i_obs, obs_noise_std = add_noise(infectious_curve, noise_level, rng)
            initial_guess = sample_initial_guess(rng)

            _, fitted_params = run_ekf(
                i_obs, dt=OBSERVATION_INTERVAL_DAYS, y0=Y0,
                param_init=initial_guess, population=POPULATION_SIZE,
                rho=1.0, observation_noise_std=obs_noise_std,
            )
            relative_error = np.abs(fitted_params - PARAMS_TRUE) / PARAMS_TRUE
            raw_results.append({
                "noise_level": noise_level,
                "seed": seed_idx,
                "beta_fitted": fitted_params[0],
                "sigma_fitted": fitted_params[1],
                "gamma_fitted": fitted_params[2],
                "beta_rel_error": relative_error[0],
                "sigma_rel_error": relative_error[1],
                "gamma_rel_error": relative_error[2],
                "mean_rel_error": float(np.mean(relative_error)),
            })

        print(f"Noise level {noise_level:.0%}: {N_SEEDS} EKF fits completed.")

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "ekf_noise_raw_results.csv"), index=False)

    summary_rows = []
    for noise_level in NOISE_LEVELS:
        subset = raw_df[raw_df["noise_level"] == noise_level]
        summary_rows.append({
            "noise_level": noise_level,
            "n_seeds": len(subset),
            "median_mean_rel_error": subset["mean_rel_error"].median(),
            "q25_mean_rel_error": subset["mean_rel_error"].quantile(0.25),
            "q75_mean_rel_error": subset["mean_rel_error"].quantile(0.75),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "ekf_noise_summary.csv"), index=False)

    print("\nSummary (EKF, noise robustness):")
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["noise_level"] * 100, summary_df["median_mean_rel_error"],
            marker="^", color="seagreen", label="Median relative error (EKF)")
    ax.fill_between(
        summary_df["noise_level"] * 100,
        summary_df["q25_mean_rel_error"],
        summary_df["q75_mean_rel_error"],
        alpha=0.2, color="seagreen", label="Interquartile range",
    )
    ax.set_xlabel("Relative observation noise level (%)")
    ax.set_ylabel("Mean relative parameter-recovery error")
    ax.set_title("SEIR parameter recovery under observation noise -- EKF baseline")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "ekf_noise_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
