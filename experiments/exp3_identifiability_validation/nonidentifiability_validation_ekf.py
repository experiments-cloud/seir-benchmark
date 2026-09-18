"""
nonidentifiability_validation_ekf.py

Validation of the practical-identifiability diagnostic (companion script:
seir_identifiability_analysis.py) against actual parameter-recovery
behavior -- Extended Kalman Filter (EKF) baseline.

Mirrors nonidentifiability_validation_nls.py and _einn.py exactly: the
same two underreporting scenarios (rho=0.30, practically identifiable;
rho=1e-5, practically non-identifiable), the same data volume, noise
level, and seeds. The EKF observes rho*I(t), the same observation model
already used by default in sparsity_benchmark_ekf.py and
noise_benchmark_ekf.py -- no change to the filter's state-transition
Jacobian is required for this experiment, since the Jacobian depends only
on the SEIR dynamics, not on the observation operator; only the
observation matrix H (already a configurable argument of run_ekf) needs
to select and scale the infectious compartment by the scenario's rho.
See seir_identifiability_analysis.py's methods description for the
rationale behind the two specific rho values.

HOW TO RUN
-----------
    pip install -r requirements.txt
    python nonidentifiability_validation_ekf.py

Outputs are written to ./results/.

DEPENDENCIES
-------------
numpy, matplotlib, pandas, and the companion modules seir_model.py and
ekf_seir.py.

LICENSE
--------
This script is intended to be released alongside the associated publication
under an open license (e.g., MIT) to support reproducibility. Replace this
notice with the license terms selected for the accompanying code repository.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seir_model import determine_outbreak_duration, solve_seir
from ekf_seir import run_ekf

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

POPULATION_SIZE = 1_000_000
BETA_TRUE = 0.35
SIGMA_TRUE = 1 / 5.2
GAMMA_TRUE = 1 / 10.0
PARAMS_TRUE = np.array([BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE])

E0, I0, R0 = 1, 1, 0
S0 = POPULATION_SIZE - E0 - I0 - R0
Y0 = np.array([S0, E0, I0, R0], dtype=float)

# Same two scenarios as the companion NLS/EINN identifiability-validation
# scripts.
SCENARIOS = {
    "control_identifiable_rho030": 0.30,
    "test_nonidentifiable_rho1e-5": 0.00001,
}

NOISE_LEVEL = 0.10
NOISE_FLOOR_CASES = 1.0
N_SEEDS = 15
OBSERVATION_INTERVAL_DAYS = 1.0

PARAM_BOUNDS_LOWER = np.array([0.01, 1 / 14, 1 / 21])
PARAM_BOUNDS_UPPER = np.array([2.00, 1 / 2, 1 / 3])

RANDOM_SEED_BASE = 12345


def generate_full_trajectory():
    full_duration = determine_outbreak_duration(PARAMS_TRUE, Y0, POPULATION_SIZE)
    n_points = int(np.ceil(full_duration / OBSERVATION_INTERVAL_DAYS)) + 1
    time_grid = np.linspace(0, full_duration, n_points)
    trajectory = solve_seir(PARAMS_TRUE, time_grid, Y0, POPULATION_SIZE)
    return time_grid, trajectory, full_duration


def add_noise(observed_true, noise_level, rng):
    noise_std = np.maximum(noise_level * np.abs(observed_true), NOISE_FLOOR_CASES)
    noisy = observed_true + rng.normal(0.0, noise_std)
    return np.clip(noisy, 0.0, None)


def sample_initial_guess(rng):
    log_lower = np.log(PARAM_BOUNDS_LOWER)
    log_upper = np.log(PARAM_BOUNDS_UPPER)
    return np.exp(rng.uniform(log_lower, log_upper))


def main():
    time_grid, full_trajectory, full_duration = generate_full_trajectory()
    infectious_curve = full_trajectory[2]

    print(f"Full outbreak duration: {full_duration:.1f} days "
          f"({len(time_grid)} daily observations).")

    raw_results = []

    for scenario_name, rho in SCENARIOS.items():
        true_observed = rho * infectious_curve
        obs_noise_std = np.maximum(NOISE_LEVEL * np.abs(true_observed), NOISE_FLOOR_CASES)

        for seed_idx in range(N_SEEDS):
            rng = np.random.default_rng(RANDOM_SEED_BASE + seed_idx)
            observed_noisy = add_noise(true_observed, NOISE_LEVEL, rng)
            initial_guess = sample_initial_guess(rng)

            _, fitted_params = run_ekf(
                observed_noisy, dt=OBSERVATION_INTERVAL_DAYS, y0=Y0,
                param_init=initial_guess, population=POPULATION_SIZE,
                rho=rho, observation_noise_std=obs_noise_std,
            )
            relative_error = np.abs(fitted_params - PARAMS_TRUE) / PARAMS_TRUE
            raw_results.append({
                "scenario": scenario_name,
                "rho": rho,
                "seed": seed_idx,
                "beta_fitted": fitted_params[0],
                "sigma_fitted": fitted_params[1],
                "gamma_fitted": fitted_params[2],
                "beta_rel_error": relative_error[0],
                "sigma_rel_error": relative_error[1],
                "gamma_rel_error": relative_error[2],
                "mean_rel_error": float(np.mean(relative_error)),
            })

        print(f"Scenario {scenario_name} (rho={rho}): {N_SEEDS} EKF fits completed.")

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "nonid_ekf_raw_results.csv"), index=False)

    summary_rows = []
    for scenario_name in SCENARIOS:
        subset = raw_df[raw_df["scenario"] == scenario_name]
        summary_rows.append({
            "scenario": scenario_name,
            "rho": subset["rho"].iloc[0],
            "n_seeds": len(subset),
            "median_beta_rel_error": subset["beta_rel_error"].median(),
            "median_sigma_rel_error": subset["sigma_rel_error"].median(),
            "median_gamma_rel_error": subset["gamma_rel_error"].median(),
            "median_mean_rel_error": subset["mean_rel_error"].median(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "nonid_ekf_summary.csv"), index=False)

    print("\nSummary (EKF, identifiability validation):")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
