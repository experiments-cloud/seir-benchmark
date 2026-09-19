"""
nonidentifiability_validation_mcmc.py

Validation of the practical-identifiability diagnostic (companion script:
seir_identifiability_analysis.py) against actual parameter-recovery
behavior, Bayesian MCMC baseline (PyMC, NUTS sampler).

Unlike the NLS, EKF, and EINN companion scripts, this script does not
replicate the full data-sparsity and observation-noise benchmark designs
(5 sparsity levels x 15 seeds, 7 noise levels x 15 seeds). Without a
compiled ODE integrator (the `sunode` package, which would let PyMC
integrate the SEIR system efficiently, requires Conda and was not
available in the development environment), PyMC's built-in ODE solver
(`pm.ode.DifferentialEquation`) integrates via SciPy on every gradient
evaluation, roughly 1 to 2 seconds per MCMC draw in development testing,
dramatically slower than the classical or neural-network-based methods.
Replicating the full benchmark design at that speed would take on the
order of days to weeks of continuous computation. Given that constraint,
MCMC is applied here only to the identifiability validation experiment
(the same two scenarios as nonidentifiability_validation_nls.py and
_einn.py: an identifiable control, rho=0.30, and a non-identifiable test
case, rho=1e-5), with a reduced number of seeds (5, not 15). This is the
single most informative place to spend MCMC's computational budget, since
MCMC is the only one of the four methods that incorporates explicit prior
information, so this experiment directly tests whether an informative
prior can partially rescue parameter recovery in the regime where the
other methods fail.

Because each individual fit can take tens of minutes, progress is saved
to disk after every completed (scenario, seed) fit. If interrupted and
re-run, the script reads its own checkpoint file, skips any (scenario,
seed) combination already completed, and continues with the rest.

Each fit takes roughly 15 to 25 minutes (2 chains, moderate tuning);
budget several hours for the full 10 fits (2 scenarios x 5 seeds), and
consider running this separately or overnight rather than as part of the
main pipeline. Outputs are written to ./results/. Requires numpy, pandas,
matplotlib, pymc, arviz, and seir_model.py.
"""

import os
import time
from datetime import datetime

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from seir_model import determine_outbreak_duration, solve_seir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "nonid_mcmc_raw_results.csv")

POPULATION_SIZE = 1_000_000
BETA_TRUE = 0.35
SIGMA_TRUE = 1 / 5.2
GAMMA_TRUE = 1 / 10.0
PARAMS_TRUE = np.array([BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE])

E0, I0, R0 = 1, 1, 0
S0 = POPULATION_SIZE - E0 - I0 - R0
Y0 = np.array([S0, E0, I0, R0], dtype=float)

# Same two scenarios as the companion NLS/EINN identifiability-validation
# scripts. See nonidentifiability_validation_nls.py for the rationale
# behind these specific rho values.
SCENARIOS = {
    "control_identifiable_rho030": 0.30,
    "test_nonidentifiable_rho1e-5": 0.00001,
}

NOISE_LEVEL = 0.10
NOISE_FLOOR_CASES = 1.0

# Reduced from 15 (used by the other methods) to 5, given the per-fit cost
# -- see the module docstring for the rationale.
N_SEEDS = 5

# Reduced number of time points at which the ODE is evaluated during
# sampling (independent of the daily-resolution data used elsewhere): each
# additional point adds directly to the cost of every gradient evaluation
# under PyMC's SciPy-based ODE integration, so this is kept deliberately
# small. The points are still spread across the full epidemic cycle.
N_TIME_POINTS_MCMC = 15

# Scaling factor applied to compartments before sampling, to keep values
# within a numerically well-conditioned range for the sampler.
STATE_SCALE = 1e5

N_TUNE = 300
N_DRAWS = 300
N_CHAINS = 2
TARGET_ACCEPT = 0.95

RANDOM_SEED_BASE = 12345


# ---------------------------------------------------------------------------
# Synthetic ground truth, observation subsampling, and noise model
# ---------------------------------------------------------------------------

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
    return [
        -beta * S * I / N,
        beta * S * I / N - sigma * E,
        sigma * E - gamma * I,
        gamma * I,
    ]


# ---------------------------------------------------------------------------
# MCMC fit for a single (scenario, seed) configuration
# ---------------------------------------------------------------------------

def fit_mcmc(time_grid, observed, rho, seed):
    y0_scaled = Y0 / STATE_SCALE
    observed_scaled = observed / (rho * STATE_SCALE)

    ode_model = pm.ode.DifferentialEquation(
        func=seir_rhs_symbolic, times=time_grid, n_states=4, n_theta=3, t0=0
    )

    with pm.Model():
        beta = pm.TruncatedNormal(
            "beta", mu=0.3, sigma=0.3, lower=0.01, upper=2.0, initval=0.3
        )
        sigma_p = pm.TruncatedNormal(
            "sigma_p", mu=0.2, sigma=0.15, lower=1 / 14, upper=1 / 2, initval=0.2
        )
        gamma = pm.TruncatedNormal(
            "gamma", mu=0.1, sigma=0.08, lower=1 / 21, upper=1 / 3, initval=0.1
        )

        ode_solution = ode_model(y0=y0_scaled, theta=[beta, sigma_p, gamma])
        i_pred_scaled = ode_solution[:, 2]

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        pm.Normal("obs", mu=i_pred_scaled, sigma=sigma_obs, observed=observed_scaled)

        idata = pm.sample(
            draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, cores=1,
            target_accept=TARGET_ACCEPT, random_seed=seed,
            progressbar=True,
        )

    posterior_means = np.array([
        float(idata.posterior["beta"].mean()),
        float(idata.posterior["sigma_p"].mean()),
        float(idata.posterior["gamma"].mean()),
    ])
    r_hat_values = {
        var: float(pm.stats.rhat(idata, var_names=[var])[var].values)
        for var in ["beta", "sigma_p", "gamma"]
    }
    ess_bulk_values = {
        var: float(az.ess(idata, var_names=[var], method="bulk")[var].values)
        for var in ["beta", "sigma_p", "gamma"]
    }
    n_divergences = int(idata.sample_stats["diverging"].sum())
    return posterior_means, r_hat_values, ess_bulk_values, n_divergences


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return pd.read_csv(CHECKPOINT_PATH)
    return pd.DataFrame(columns=[
        "scenario", "rho", "seed", "beta_fitted", "sigma_fitted",
        "gamma_fitted", "beta_rel_error", "sigma_rel_error",
        "gamma_rel_error", "mean_rel_error", "rhat_beta", "rhat_sigma",
        "rhat_gamma", "ess_bulk_beta", "ess_bulk_sigma", "ess_bulk_gamma",
        "n_divergences", "fit_duration_seconds", "completed_at",
    ])


def append_checkpoint(row_dict):
    df_row = pd.DataFrame([row_dict])
    write_header = not os.path.exists(CHECKPOINT_PATH)
    df_row.to_csv(CHECKPOINT_PATH, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def main():
    time_grid, full_trajectory = generate_observation_grid()
    infectious_curve = full_trajectory[2]

    all_jobs = [
        (scenario_name, rho, seed_idx)
        for scenario_name, rho in SCENARIOS.items()
        for seed_idx in range(N_SEEDS)
    ]
    total_jobs = len(all_jobs)

    checkpoint_df = load_checkpoint()
    completed_pairs = set(
        zip(checkpoint_df["scenario"], checkpoint_df["seed"])
    ) if len(checkpoint_df) else set()

    print(f"Total fits required: {total_jobs}. "
          f"Already completed (from checkpoint): {len(completed_pairs)}.")

    for job_index, (scenario_name, rho, seed_idx) in enumerate(all_jobs, start=1):
        if (scenario_name, seed_idx) in completed_pairs:
            print(f"[{job_index}/{total_jobs}] SKIP (already done): "
                  f"scenario={scenario_name}, seed={seed_idx}")
            continue

        print(f"[{job_index}/{total_jobs}] START "
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
              f"scenario={scenario_name} (rho={rho}), seed={seed_idx}")
        fit_start = time.time()

        true_observed = rho * infectious_curve
        rng = np.random.default_rng(RANDOM_SEED_BASE + seed_idx)
        observed_noisy = add_noise(true_observed, NOISE_LEVEL, rng)

        try:
            fitted_params, r_hat_values, ess_bulk_values, n_divergences = fit_mcmc(
                time_grid, observed_noisy, rho, seed=RANDOM_SEED_BASE + seed_idx
            )
            relative_error = np.abs(fitted_params - PARAMS_TRUE) / PARAMS_TRUE
            fit_duration = time.time() - fit_start

            row = {
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
                "rhat_beta": r_hat_values["beta"],
                "rhat_sigma": r_hat_values["sigma_p"],
                "rhat_gamma": r_hat_values["gamma"],
                "ess_bulk_beta": ess_bulk_values["beta"],
                "ess_bulk_sigma": ess_bulk_values["sigma_p"],
                "ess_bulk_gamma": ess_bulk_values["gamma"],
                "n_divergences": n_divergences,
                "fit_duration_seconds": fit_duration,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_checkpoint(row)

            print(f"[{job_index}/{total_jobs}] DONE in {fit_duration:.0f}s -- "
                  f"beta={fitted_params[0]:.4f} sigma={fitted_params[1]:.4f} "
                  f"gamma={fitted_params[2]:.4f} "
                  f"(r_hat: {r_hat_values['beta']:.3f}/"
                  f"{r_hat_values['sigma_p']:.3f}/{r_hat_values['gamma']:.3f}) "
                  f"-- checkpoint saved.")
        except Exception as e:
            print(f"[{job_index}/{total_jobs}] FAILED: {scenario_name}, "
                  f"seed={seed_idx} -- {e}. Continuing with next fit; "
                  f"this one will be retried on the next run.")

    # Once all jobs are done (or on partial re-runs), (re)build the
    # summary table and plot from whatever is in the checkpoint file.
    final_df = load_checkpoint()
    if len(final_df) == 0:
        print("No completed fits yet -- nothing to summarize.")
        return

    summary_rows = []
    for scenario_name, rho in SCENARIOS.items():
        subset = final_df[final_df["scenario"] == scenario_name]
        if len(subset) == 0:
            continue
        summary_rows.append({
            "scenario": scenario_name,
            "rho": rho,
            "n_seeds_completed": len(subset),
            "n_seeds_target": N_SEEDS,
            "median_beta_rel_error": subset["beta_rel_error"].median(),
            "median_sigma_rel_error": subset["sigma_rel_error"].median(),
            "median_gamma_rel_error": subset["gamma_rel_error"].median(),
            "median_mean_rel_error": subset["mean_rel_error"].median(),
            "mean_rhat_beta": subset["rhat_beta"].mean(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "nonid_mcmc_summary.csv"),
                       index=False)

    print("\nSummary (MCMC, identifiability validation -- based on fits "
          "completed so far):")
    print(summary_df.to_string(index=False))

    if len(summary_df) == len(SCENARIOS) and \
            (summary_df["n_seeds_completed"] == N_SEEDS).all():
        fig, ax = plt.subplots(figsize=(7.5, 5))
        x = np.arange(len(SCENARIOS))
        width = 0.25
        colors = ["#4c72b0", "#dd8452", "#55a868"]
        for i, param in enumerate(["beta", "sigma", "gamma"]):
            values = summary_df[f"median_{param}_rel_error"]
            ax.bar(x + i * width, values, width, label=param, color=colors[i])
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{s}\n(rho={r})" for s, r in
                             zip(summary_df["scenario"], summary_df["rho"])])
        ax.set_ylabel("Median relative parameter-recovery error")
        ax.set_title("MCMC: parameter recovery in predicted identifiable\n"
                      "vs. non-identifiable regimes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "nonid_mcmc_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"\nAll fits complete. Plot saved to {OUTPUT_DIR}/nonid_mcmc_comparison.png")
    else:
        print("\nNot all fits are complete yet -- re-run this script to "
              "continue from the checkpoint. Plot will be generated once "
              "all fits are done.")


if __name__ == "__main__":
    main()
