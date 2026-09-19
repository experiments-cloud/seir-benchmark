"""
real_data_fit_mcmc.py

Real-world validation of the SEIR parameter-recovery pipeline, Bayesian
MCMC baseline (PyMC, NUTS sampler). Fits (beta, sigma, gamma) to real
first-wave COVID-19 cumulative case data for Italy and South Korea, with
the reporting fraction rho fixed at the NLS-estimated value (see
real_data_fit_ekf.py for the same design choice and its rationale).

The full synthetic sparsity/noise MCMC benchmark (matching the NLS/EKF/
EINN design of 15 seeds across several conditions) was judged
computationally infeasible in this environment, on the order of 9 days of
continuous computation; see nonidentifiability_validation_mcmc.py. A
single real-data validation, two fits total, one per country, has a very
different cost profile and is worth running on its own: the synthetic
identifiability-validation experiment found that MCMC's informative
priors substantially rescue parameter recovery when the reporting
fraction is very small, and underreporting in real case-count data is a
practical instance of the same identifiability problem, so testing
whether that advantage carries over to real, underreported data connects
the study's most promising synthetic result to real-world data directly.

As in nonidentifiability_validation_mcmc.py, progress is saved after each
completed country fit, so an interruption only costs the fit in progress.

Run download_real_data.py and real_data_fit_nls.py first (the latter
provides rho), then this script. Each country fit takes roughly 15 to 25
minutes (2 chains, 300 tuning plus 300 draws); results are written to
./results/. Requires numpy, pandas, matplotlib, pymc, arviz, and
seir_model.py.
"""

import os
import time
from datetime import datetime

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "real_data_mcmc_raw_results.csv")

COUNTRIES = {
    "italy": {"population": 60_300_000, "data_file": "real_data_italy.csv",
              "display_name": "Italy"},
    "korea": {"population": 51_780_000, "data_file": "real_data_korea.csv",
              "display_name": "South Korea"},
}

STATE_SCALE = 1e4
N_TUNE = 300
N_DRAWS = 300
N_CHAINS = 2
TARGET_ACCEPT = 0.95
RANDOM_SEED = 12345


def load_country_data_and_rho(country_key):
    config = COUNTRIES[country_key]
    df = pd.read_csv(config["data_file"])
    t_obs = df["day"].values.astype(float)
    observed = df["cumulative_confirmed"].values.astype(float)
    population = config["population"]

    i0 = observed[0]
    y0 = np.array([population - 2 * i0, i0, i0, 0.0])

    nls_summary_path = os.path.join(OUTPUT_DIR, "real_data_nls_summary.csv")
    if not os.path.exists(nls_summary_path):
        raise FileNotFoundError(
            "real_data_nls_summary.csv not found -- run real_data_fit_nls.py first."
        )
    nls_summary = pd.read_csv(nls_summary_path)
    row = nls_summary[nls_summary["country"] == config["display_name"]]
    rho = float(row["rho"].iloc[0])

    return t_obs, observed, y0, population, rho


def seir_rhs_symbolic(y, t, p, population):
    S, E, I, R = y
    beta, sigma, gamma = p[0], p[1], p[2]
    N = population / STATE_SCALE
    return [
        -beta * S * I / N,
        beta * S * I / N - sigma * E,
        sigma * E - gamma * I,
        gamma * I,
    ]


def fit_mcmc(t_obs, observed, y0, population, rho, seed):
    y0_scaled = y0 / STATE_SCALE
    observed_scaled = observed / (rho * STATE_SCALE)

    ode_model = pm.ode.DifferentialEquation(
        func=lambda y, t, p: seir_rhs_symbolic(y, t, p, population),
        times=t_obs, n_states=4, n_theta=3, t0=0,
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
        simulated_scaled = ode_solution[:, 2] + ode_solution[:, 3]

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        pm.Normal("obs", mu=simulated_scaled, sigma=sigma_obs, observed=observed_scaled)

        idata = pm.sample(
            draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, cores=1,
            target_accept=TARGET_ACCEPT, random_seed=seed, progressbar=True,
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


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return pd.read_csv(CHECKPOINT_PATH)
    return pd.DataFrame(columns=[
        "country", "beta", "sigma", "gamma", "rho", "rhat_beta",
        "rhat_sigma", "rhat_gamma", "ess_bulk_beta", "ess_bulk_sigma",
        "ess_bulk_gamma", "n_divergences", "fit_duration_seconds", "completed_at",
    ])


def append_checkpoint(row_dict):
    df_row = pd.DataFrame([row_dict])
    write_header = not os.path.exists(CHECKPOINT_PATH)
    df_row.to_csv(CHECKPOINT_PATH, mode="a", header=write_header, index=False)


def plot_fit(country_key, display_name, t_obs, observed, y0, population,
             rho, fitted_params):
    from seir_model import solve_seir
    trajectory = solve_seir(tuple(fitted_params), t_obs, y0, population)
    simulated_observed = rho * (trajectory[2] + trajectory[3])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_obs, observed, "o", markersize=3, color="steelblue",
             label="Observed (JHU CSSE)")
    ax.plot(t_obs, simulated_observed, "-", color="#8172b3", linewidth=2,
             label="MCMC posterior-mean trajectory")
    ax.set_xlabel("Days since window start")
    ax.set_ylabel("Cumulative confirmed cases")
    valid_mask = observed >= 50
    relative_residuals = ((simulated_observed - observed) / np.maximum(observed, 10.0))[valid_mask]
    rmse_relative = float(np.sqrt(np.mean(relative_residuals ** 2)))
    ax.set_title(
        f"{display_name}: MCMC fit to real first-wave data\n"
        f"beta={fitted_params[0]:.3f}, sigma={fitted_params[1]:.3f}, "
        f"gamma={fitted_params[2]:.3f}, rho={rho:.4f} (fixed, from NLS), "
        f"relative RMSE={rmse_relative:.2%}"
    )
    ax.legend()
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"real_data_fit_mcmc_{country_key}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return rmse_relative


def main():
    checkpoint_df = load_checkpoint()
    completed_countries = set(checkpoint_df["country"]) if len(checkpoint_df) else set()

    for country_key, config in COUNTRIES.items():
        if country_key in completed_countries:
            print(f"SKIP (already done): {config['display_name']}")
            continue

        print(f"\nFitting {config['display_name']} (MCMC) -- "
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
        fit_start = time.time()

        t_obs, observed, y0, population, rho = load_country_data_and_rho(country_key)

        try:
            fitted_params, r_hat_values, ess_bulk_values, n_divergences = fit_mcmc(
                t_obs, observed, y0, population, rho, seed=RANDOM_SEED
            )
            fit_duration = time.time() - fit_start

            row = {
                "country": country_key,
                "beta": fitted_params[0],
                "sigma": fitted_params[1],
                "gamma": fitted_params[2],
                "rho": rho,
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

            rmse = plot_fit(country_key, config["display_name"], t_obs, observed,
                             y0, population, rho, fitted_params)

            print(f"DONE in {fit_duration:.0f}s -- beta={fitted_params[0]:.4f} "
                  f"sigma={fitted_params[1]:.4f} gamma={fitted_params[2]:.4f} "
                  f"relative RMSE={rmse:.2%} "
                  f"(r_hat: {r_hat_values['beta']:.3f}/{r_hat_values['sigma_p']:.3f}/"
                  f"{r_hat_values['gamma']:.3f}) -- checkpoint saved.")
        except Exception as e:
            print(f"FAILED: {config['display_name']} -- {e}. "
                  f"Will be retried on the next run.")

    final_df = load_checkpoint()
    if len(final_df) > 0:
        final_df.to_csv(os.path.join(OUTPUT_DIR, "real_data_mcmc_summary.csv"),
                         index=False)
        print("\nFinal summary:")
        print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
