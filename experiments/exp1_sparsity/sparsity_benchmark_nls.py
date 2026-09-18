"""
sparsity_benchmark_nls.py

Data-sparsity benchmark for SEIR parameter recovery -- classical nonlinear
least-squares (NLS) baseline.

WHAT THIS SCRIPT DOES
----------------------
1. Generates a synthetic ground-truth SEIR outbreak (no observation noise;
   noise robustness is addressed in a separate companion experiment).
2. For each data-sparsity level (the fraction of the full epidemic cycle
   made available for fitting: 15%, 30%, 45%, 60%, 100%), fits the
   transmission rate (beta), incubation rate (sigma), and recovery rate
   (gamma) to the observed infectious curve I(t) using nonlinear
   least-squares, by numerically integrating the SEIR ODE inside the
   optimization loop.
3. Repeats each fit from multiple random initial guesses (seeds) to check
   convergence robustness, since the SEIR inverse problem is known to have
   partially confounded parameters (see companion identifiability-analysis
   script) and NLS can converge to different local solutions depending on
   initialization.
4. Reports, for each sparsity level: the median and interquartile range of
   the relative parameter-recovery error, and the fraction of seeds that
   converged to a numerically stable solution.
5. Saves raw per-run results, a summary table, and a summary plot.

This script implements ONE baseline (NLS) in an experimental design that
also includes a physics-informed neural network estimator and additional
classical baselines (extended Kalman filter, Bayesian/MCMC), released as
companion scripts, so that all methods can be benchmarked under an
identical experimental protocol.

HOW TO RUN
-----------
    pip install -r requirements.txt
    python sparsity_benchmark_nls.py

Outputs are written to ./results/ (created automatically).

DEPENDENCIES
-------------
numpy, scipy, matplotlib, pandas, and the companion module seir_model.py
(must be in the same directory or on the Python path).

LICENSE
--------
This script is intended to be released alongside the associated publication
under an open license (e.g., MIT) to support reproducibility. Replace this
notice with the license terms selected for the accompanying code repository.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from seir_model import determine_outbreak_duration, solve_seir

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
Y0 = [S0, E0, I0, R0]

# Fraction of the full epidemic cycle used for fitting. 1.00 corresponds to
# the full outbreak (growth, peak, and decline); smaller values withhold
# the later part of the curve, simulating an ongoing, unresolved outbreak.
DATA_FRACTIONS = [0.15, 0.30, 0.45, 0.60, 1.00]

# Number of random-initial-guess repetitions per sparsity level, used to
# assess convergence robustness (not to model observation noise, which is
# absent in this experiment by design).
N_SEEDS = 15

OBSERVATION_INTERVAL_DAYS = 1.0  # one observation per simulated day

# Bounds and random-initial-guess ranges for the optimizer, chosen to span
# epidemiologically plausible values without assuming knowledge of the
# ground truth.
PARAM_BOUNDS_LOWER = [0.01, 1 / 14, 1 / 21]   # beta, sigma, gamma
PARAM_BOUNDS_UPPER = [2.00, 1 / 2, 1 / 3]

RANDOM_SEED_BASE = 12345


# ---------------------------------------------------------------------------
# Synthetic ground truth
# ---------------------------------------------------------------------------

def generate_full_trajectory():
    """Simulate the full (noise-free) SEIR outbreak at daily resolution."""
    full_duration = determine_outbreak_duration(PARAMS_TRUE, Y0, POPULATION_SIZE)
    n_points = int(np.ceil(full_duration / OBSERVATION_INTERVAL_DAYS)) + 1
    time_grid = np.linspace(0, full_duration, n_points)
    trajectory = solve_seir(PARAMS_TRUE, time_grid, Y0, POPULATION_SIZE)
    return time_grid, trajectory, full_duration


# ---------------------------------------------------------------------------
# Nonlinear least-squares fitting
# ---------------------------------------------------------------------------

def residuals(log_params, t_obs, i_obs, y0, population):
    """Residual function for least_squares. Parameters are optimized in log
    space to enforce positivity without needing box constraints to bind.
    Uses looser integrator tolerances than the reference trajectory, since
    this function is evaluated many times per fit; the effect on the
    fitted-parameter accuracy is negligible relative to the identifiability
    limits already characterized in the companion FIM analysis.
    """
    params = np.exp(log_params)
    beta, sigma, gamma = params
    trajectory = solve_seir(
        (beta, sigma, gamma), t_obs, y0, population, rtol=1e-6, atol=1e-8
    )
    i_simulated = trajectory[2]
    return i_simulated - i_obs


def fit_nls(t_obs, i_obs, y0, population, initial_guess):
    """Fit (beta, sigma, gamma) to observed infectious-curve data via
    nonlinear least-squares, working in log-parameter space.
    """
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
    """Draw a random initial guess uniformly (in log space) within the
    parameter bounds.
    """
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

    for fraction in DATA_FRACTIONS:
        n_obs = max(int(np.floor(fraction * len(time_grid))), 5)
        t_obs = time_grid[:n_obs]
        i_obs = infectious_curve[:n_obs]

        rng = np.random.default_rng(RANDOM_SEED_BASE + int(fraction * 1000))

        for seed_idx in range(N_SEEDS):
            initial_guess = sample_initial_guess(rng)
            try:
                fitted_params, converged, cost = fit_nls(
                    t_obs, i_obs, Y0, POPULATION_SIZE, initial_guess
                )
                relative_error = np.abs(fitted_params - PARAMS_TRUE) / PARAMS_TRUE
                raw_results.append({
                    "fraction": fraction,
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
                    "fraction": fraction,
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

        print(f"Fraction {fraction:.0%}: {n_obs} observations, "
              f"{N_SEEDS} random-initialization fits completed.")

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "nls_raw_results.csv"), index=False)

    # Summary statistics per sparsity level
    summary_rows = []
    for fraction in DATA_FRACTIONS:
        subset = raw_df[raw_df["fraction"] == fraction]
        summary_rows.append({
            "fraction": fraction,
            "n_seeds": len(subset),
            "convergence_rate": subset["converged"].mean(),
            "median_mean_rel_error": subset["mean_rel_error"].median(),
            "q25_mean_rel_error": subset["mean_rel_error"].quantile(0.25),
            "q75_mean_rel_error": subset["mean_rel_error"].quantile(0.75),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "nls_summary.csv"), index=False)

    print("\nSummary (NLS baseline):")
    print(summary_df.to_string(index=False))

    # Summary plot: median relative error vs. data fraction, with IQR band
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["fraction"] * 100, summary_df["median_mean_rel_error"],
            marker="o", label="Median relative error (NLS)")
    ax.fill_between(
        summary_df["fraction"] * 100,
        summary_df["q25_mean_rel_error"],
        summary_df["q75_mean_rel_error"],
        alpha=0.2, label="Interquartile range",
    )
    ax.set_xlabel("Fraction of the epidemic curve used for fitting (%)")
    ax.set_ylabel("Mean relative parameter-recovery error")
    ax.set_title("SEIR parameter recovery under data sparsity -- NLS baseline")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "nls_sparsity_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
