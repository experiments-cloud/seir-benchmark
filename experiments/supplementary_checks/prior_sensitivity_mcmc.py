"""
prior_sensitivity_mcmc.py

Prior sensitivity check for the MCMC identifiability-validation experiment
(nonidentifiability_validation_mcmc.py): repeats the non-identifiable
scenario (rho = 1e-5) under three prior widths, half, the original width
used throughout the study (Methods, Section 3.4), and double, to check
whether MCMC's advantage in that scenario depends on how informative the
chosen prior happens to be. Only the non-identifiable scenario is run,
since that is where the prior does the most work and where a reviewer
would most reasonably ask whether the result is prior-dependent; the
control (identifiable) scenario is not repeated here, since the
likelihood already dominates there regardless of prior width.

Two random seeds per prior width (six fits total). Each fit uses the same
tune/draws/target_accept as the rest of this study's MCMC experiments
(300 tuning, 300 post-tuning draws, two chains, target_accept=0.95), so
expect roughly the same per-fit runtime as the original identifiability
validation experiment, on the order of 15 to 25 minutes each; budget two
to three hours for all six fits, or run them across several sessions,
since results are checkpointed to results_prior_sensitivity/
prior_sensitivity_raw_results.csv after every completed fit and the
script resumes automatically from there if interrupted.

Requires seir_model.py (included alongside this script) and the same
Python environment as the rest of this study (numpy, pandas, pymc).
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
from seir_model import determine_outbreak_duration, solve_seir

OUTPUT_DIR = "results_prior_sensitivity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

POPULATION_SIZE = 1_000_000
BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE = 0.35, 1/5.2, 1/10.0
PARAMS_TRUE = np.array([BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE])
E0, I0, R0 = 1, 1, 0
S0 = POPULATION_SIZE - E0 - I0 - R0
Y0 = np.array([S0, E0, I0, R0], dtype=float)

RHO_TEST = 0.00001  # el escenario no identificable, rho=1e-5
NOISE_LEVEL = 0.10
NOISE_FLOOR_CASES = 1.0
N_TIME_POINTS_MCMC = 15
STATE_SCALE = 1e5
N_TUNE = 300
N_DRAWS = 300
N_CHAINS = 2
TARGET_ACCEPT = 0.95
RANDOM_SEED_BASE = 12345

# Ancho original (Methods 3.4) vs. dos veces mas ancha y dos veces mas angosta
PRIOR_WIDTHS = {
    "narrow_0.5x": {"beta_sigma": 0.15, "sigma_p_sigma": 0.075, "gamma_sigma": 0.04},
    "original_1x": {"beta_sigma": 0.30, "sigma_p_sigma": 0.150, "gamma_sigma": 0.08},
    "wide_2x":     {"beta_sigma": 0.60, "sigma_p_sigma": 0.300, "gamma_sigma": 0.16},
}
N_SEEDS = 2

def generate_observation_grid():
    full_duration = determine_outbreak_duration(PARAMS_TRUE, Y0, POPULATION_SIZE)
    time_grid = np.linspace(0, full_duration, N_TIME_POINTS_MCMC)
    trajectory = solve_seir(PARAMS_TRUE, time_grid, Y0, POPULATION_SIZE)
    return time_grid, trajectory

def add_noise(observed_true, noise_level, rng):
    noise_std = np.maximum(noise_level * np.abs(observed_true), NOISE_FLOOR_CASES)
    noisy = observed_true + rng.normal(0.0, noise_std)
    return np.clip(noisy, 0.0, None)

def seir_rhs_symbolic(y, t, p):
    S, E, I, R = y
    beta, sigma, gamma = p[0], p[1], p[2]
    N = POPULATION_SIZE / STATE_SCALE
    return [-beta*S*I/N, beta*S*I/N - sigma*E, sigma*E - gamma*I, gamma*I]

def fit_mcmc(time_grid, observed, rho, prior_widths, seed):
    y0_scaled = Y0 / STATE_SCALE
    observed_scaled = observed / (rho * STATE_SCALE)

    ode_model = pm.ode.DifferentialEquation(
        func=seir_rhs_symbolic, times=time_grid, n_states=4, n_theta=3, t0=0
    )

    with pm.Model():
        beta = pm.TruncatedNormal("beta", mu=0.3, sigma=prior_widths["beta_sigma"],
                                    lower=0.01, upper=2.0, initval=0.3)
        sigma_p = pm.TruncatedNormal("sigma_p", mu=0.2, sigma=prior_widths["sigma_p_sigma"],
                                       lower=1/14, upper=1/2, initval=0.2)
        gamma = pm.TruncatedNormal("gamma", mu=0.1, sigma=prior_widths["gamma_sigma"],
                                     lower=1/21, upper=1/3, initval=0.1)

        ode_solution = ode_model(y0=y0_scaled, theta=[beta, sigma_p, gamma])
        i_pred_scaled = ode_solution[:, 2]
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        pm.Normal("obs", mu=i_pred_scaled, sigma=sigma_obs, observed=observed_scaled)

        idata = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, cores=1,
                           target_accept=TARGET_ACCEPT, random_seed=seed, progressbar=True)

    posterior_means = np.array([
        float(idata.posterior["beta"].mean()),
        float(idata.posterior["sigma_p"].mean()),
        float(idata.posterior["gamma"].mean()),
    ])
    r_hat_values = {v: float(pm.stats.rhat(idata, var_names=[v])[v].values)
                     for v in ["beta", "sigma_p", "gamma"]}
    return posterior_means, r_hat_values

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "prior_sensitivity_raw_results.csv")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return pd.read_csv(CHECKPOINT_PATH)
    return pd.DataFrame(columns=["prior_width", "seed", "beta_fitted", "sigma_fitted",
                                   "gamma_fitted", "mean_rel_error", "rhat_beta", "rhat_sigma", "rhat_gamma"])

def append_checkpoint(row):
    df = pd.DataFrame([row])
    write_header = not os.path.exists(CHECKPOINT_PATH)
    df.to_csv(CHECKPOINT_PATH, mode="a", header=write_header, index=False)

def main(max_fits_this_call=2):
    time_grid, full_trajectory = generate_observation_grid()
    infectious_curve = full_trajectory[2]
    true_observed = RHO_TEST * infectious_curve

    checkpoint = load_checkpoint()
    done_pairs = set(zip(checkpoint["prior_width"], checkpoint["seed"])) if len(checkpoint) else set()

    all_jobs = [(pw, seed) for pw in PRIOR_WIDTHS for seed in range(N_SEEDS)]
    n_done_this_call = 0
    for prior_width_name, seed_idx in all_jobs:
        if (prior_width_name, seed_idx) in done_pairs:
            continue
        if n_done_this_call >= max_fits_this_call:
            print(f"Reached this call's budget; {len(done_pairs)+n_done_this_call}/{len(all_jobs)} total done.")
            return
        rng = np.random.default_rng(RANDOM_SEED_BASE + seed_idx)
        observed_noisy = add_noise(true_observed, NOISE_LEVEL, rng)
        fitted, r_hat = fit_mcmc(time_grid, observed_noisy, RHO_TEST,
                                   PRIOR_WIDTHS[prior_width_name], seed=RANDOM_SEED_BASE+seed_idx)
        rel_err = np.abs(fitted - PARAMS_TRUE) / PARAMS_TRUE
        row = {"prior_width": prior_width_name, "seed": seed_idx,
               "beta_fitted": fitted[0], "sigma_fitted": fitted[1], "gamma_fitted": fitted[2],
               "mean_rel_error": float(np.mean(rel_err)),
               "rhat_beta": r_hat["beta"], "rhat_sigma": r_hat["sigma_p"], "rhat_gamma": r_hat["gamma"]}
        append_checkpoint(row)
        n_done_this_call += 1
        print(f"DONE {prior_width_name} seed={seed_idx}: mean_rel_error={row['mean_rel_error']:.4f} "
              f"rhat=({r_hat['beta']:.3f}/{r_hat['sigma_p']:.3f}/{r_hat['gamma']:.3f}) "
              f"({len(done_pairs)+n_done_this_call}/{len(all_jobs)} total)")

    print(f"ALL {len(all_jobs)} FITS COMPLETE.")

if __name__ == "__main__":
    main(max_fits_this_call=int(os.environ.get("MAX_FITS", 2)))
