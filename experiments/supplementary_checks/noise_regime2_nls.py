"""
noise_benchmark_nls.py

Observation-noise robustness benchmark for SEIR parameter recovery --
classical nonlinear least-squares (NLS) baseline.

1. Generates the same synthetic ground-truth SEIR outbreak used by the
   companion data-sparsity experiment, observed over the FULL epidemic
   cycle (this experiment isolates the effect of observation noise from
   the effect of data sparsity, which is characterized separately).
2. For each noise level (0%, 5%, 15%, 30% relative Gaussian noise added to
   the observed infectious curve), fits (beta, sigma, gamma) via
   nonlinear least-squares against the noisy observations.
3. Repeats each fit across multiple random noise realizations AND random
   optimizer initial guesses (one combined seed controls both, for
   reproducibility), to characterize both noise sensitivity and
   convergence robustness jointly.
4. Reports, for each noise level: the median and interquartile range of
   the relative parameter-recovery error.
5. Saves raw per-run results, a summary table, and a summary plot.

The noise model matches the one used in the companion identifiability
(Fisher Information Matrix) analysis: Gaussian noise with standard
deviation proportional to the magnitude of each observation (heteroscedastic,
a common choice for epidemic case-count data), with a floor to avoid
zero-variance noise at very low counts. Negative noisy observations are
clipped to zero, since case counts cannot be negative.


Outputs are written to ./results/ (created automatically).

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from seir_model import determine_outbreak_duration, solve_seir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results_noise_regime2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

POPULATION_SIZE = 1_000_000
BETA_TRUE = 0.9
SIGMA_TRUE = 1 / 8.0
GAMMA_TRUE = 1 / 10.0
PARAM_NAMES = ["beta", "sigma", "gamma"]
PARAMS_TRUE = np.array([BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE])

E0, I0, R0 = 1, 1, 0
S0 = POPULATION_SIZE - E0 - I0 - R0
Y0 = [S0, E0, I0, R0]

# Relative observation-noise levels (0.0 included as a noise-free
# reference point, matching the fraction=1.00 case of the companion
# sparsity experiment). The range extends beyond the levels originally
# specified in the experimental protocol (0/5/15/30%) up to 100%, after
# preliminary (small-sample) testing during development suggested the two
# methods' error curves may cross somewhere in the 30-70% range -- this
# extended range is intended to locate that crossover, if it exists, with
# the full seed count rather than leaving it uncharacterized.
NOISE_LEVELS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 1.00]
NOISE_FLOOR_CASES = 1.0  # minimum absolute noise standard deviation, in cases

N_SEEDS = 5
OBSERVATION_INTERVAL_DAYS = 1.0

PARAM_BOUNDS_LOWER = [0.01, 1 / 14, 1 / 21]
PARAM_BOUNDS_UPPER = [2.00, 1 / 2, 1 / 3]

RANDOM_SEED_BASE = 12345


# ---------------------------------------------------------------------------
# Synthetic ground truth and noise model
# ---------------------------------------------------------------------------

def generate_full_trajectory():
    full_duration = determine_outbreak_duration(PARAMS_TRUE, Y0, POPULATION_SIZE)
    n_points = int(np.ceil(full_duration / OBSERVATION_INTERVAL_DAYS)) + 1
    time_grid = np.linspace(0, full_duration, n_points)
    trajectory = solve_seir(PARAMS_TRUE, time_grid, Y0, POPULATION_SIZE)
    return time_grid, trajectory, full_duration


def add_noise(i_true, noise_level, rng):
    """Add heteroscedastic Gaussian noise, proportional to the true
    observation magnitude (with a floor), matching the noise model used in
    the companion identifiability analysis. Negative values are clipped to
    zero (case counts cannot be negative).
    """
    if noise_level == 0.0:
        return i_true.copy()
    noise_std = np.maximum(noise_level * np.abs(i_true), NOISE_FLOOR_CASES)
    noisy = i_true + rng.normal(0.0, noise_std)
    return np.clip(noisy, 0.0, None)


# ---------------------------------------------------------------------------
# Nonlinear least-squares fitting (identical procedure to the companion
# sparsity-benchmark script)
# ---------------------------------------------------------------------------

def residuals(log_params, t_obs, i_obs, y0, population):
    params = np.exp(log_params)
    beta, sigma, gamma = params
    trajectory = solve_seir(
        (beta, sigma, gamma), t_obs, y0, population, rtol=1e-6, atol=1e-8
    )
    return trajectory[2] - i_obs


def fit_nls(t_obs, i_obs, y0, population, initial_guess):
    log_lower = np.log(PARAM_BOUNDS_LOWER)
    log_upper = np.log(PARAM_BOUNDS_UPPER)
    log_initial = np.log(initial_guess)

    result = least_squares(
        residuals,
        log_initial,
        bounds=(log_lower, log_upper),
        args=(t_obs, i_obs, y0, population),
        method="trf",
        max_nfev=500,
    )
    fitted_params = np.exp(result.x)
    return fitted_params, result.success, result.cost


def sample_initial_guess(rng):
    log_lower = np.log(PARAM_BOUNDS_LOWER)
    log_upper = np.log(PARAM_BOUNDS_UPPER)
    log_guess = rng.uniform(log_lower, log_upper)
    return np.exp(log_guess)


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
            i_obs = add_noise(infectious_curve, noise_level, rng)
            initial_guess = sample_initial_guess(rng)

            try:
                fitted_params, converged, cost = fit_nls(
                    time_grid, i_obs, Y0, POPULATION_SIZE, initial_guess
                )
                relative_error = np.abs(fitted_params - PARAMS_TRUE) / PARAMS_TRUE
                raw_results.append({
                    "noise_level": noise_level,
                    "seed": seed_idx,
                    "converged": bool(converged),
                    "cost": float(cost),
                    "beta_fitted": fitted_params[0],
                    "sigma_fitted": fitted_params[1],
                    "gamma_fitted": fitted_params[2],
                    "beta_rel_error": relative_error[0],
                    "sigma_rel_error": relative_error[1],
                    "gamma_rel_error": relative_error[2],
                    "mean_rel_error": float(np.mean(relative_error)),
                })
            except RuntimeError as e:
                raw_results.append({
                    "noise_level": noise_level,
                    "seed": seed_idx,
                    "converged": False,
                    "cost": np.nan,
                    "beta_fitted": np.nan,
                    "sigma_fitted": np.nan,
                    "gamma_fitted": np.nan,
                    "beta_rel_error": np.nan,
                    "sigma_rel_error": np.nan,
                    "gamma_rel_error": np.nan,
                    "mean_rel_error": np.nan,
                    "error_message": str(e),
                })

        print(f"Noise level {noise_level:.0%}: {N_SEEDS} fits completed.")

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "nls_noise_raw_results.csv"), index=False)

    summary_rows = []
    for noise_level in NOISE_LEVELS:
        subset = raw_df[raw_df["noise_level"] == noise_level]
        summary_rows.append({
            "noise_level": noise_level,
            "n_seeds": len(subset),
            "convergence_rate": subset["converged"].mean(),
            "median_mean_rel_error": subset["mean_rel_error"].median(),
            "q25_mean_rel_error": subset["mean_rel_error"].quantile(0.25),
            "q75_mean_rel_error": subset["mean_rel_error"].quantile(0.75),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "nls_noise_summary.csv"), index=False)

    print("\nSummary (NLS, noise robustness):")
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["noise_level"] * 100, summary_df["median_mean_rel_error"],
            marker="o", label="Median relative error (NLS)")
    ax.fill_between(
        summary_df["noise_level"] * 100,
        summary_df["q25_mean_rel_error"],
        summary_df["q75_mean_rel_error"],
        alpha=0.2, label="Interquartile range",
    )
    ax.set_xlabel("Relative observation noise level (%)")
    ax.set_ylabel("Mean relative parameter-recovery error")
    ax.set_title("SEIR parameter recovery under observation noise -- NLS baseline")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "nls_noise_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
