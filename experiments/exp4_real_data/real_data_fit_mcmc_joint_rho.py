"""
real_data_fit_mcmc_joint_rho.py

Robustness check for real_data_fit_mcmc.py: instead of fixing the
reporting fraction rho at the value already estimated by NLS, this
version estimates rho jointly with (beta, sigma, gamma) as a fourth
Bayesian parameter, removing the asymmetry (relative to NLS and EINN)
noted as a limitation in the main analysis.

rho can span several orders of magnitude (from a few hundredths to a few
ten-thousandths in this study's NLS estimates), so a direct Uniform(0,1)
prior would concentrate almost no mass near the plausible region and
sample poorly. Instead the prior is placed on log(rho),
Normal(log(0.01), 3), weakly informative and spanning several orders of
magnitude on either side of 1%.

The original fixed-rho script divided the observed data by rho to bring
it onto the same scale as the ODE solution, which is safe for a fixed
constant but not for a sampled random variable that can take values
close to zero (dividing by a near-zero sampled quantity produces extreme
likelihood gradients and destabilizes the sampler). This version instead
multiplies the simulated trajectory by rho and compares directly to the
fixed-scale observed data, avoiding division by a sampled quantity.

Outputs are written to ./results/. Does not require real_data_fit_nls.py
to have been run first, unlike the fixed-rho version, since rho is no
longer taken from it.
"""

import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "real_data_mcmc_jointrho_raw_results.csv")

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
TARGET_ACCEPT = 0.99  # raised from 0.95 after validating that it eliminates
                       # the sampling divergences observed at the lower
                       # setting (see the manuscript's Limitations section);
                       # this increases the per-draw cost substantially.
RANDOM_SEED = 12345

LOG_RHO_PRIOR_MU = np.log(0.01)
LOG_RHO_PRIOR_SIGMA = 3.0


def load_country_data(country_key):
    config = COUNTRIES[country_key]
    df = pd.read_csv(config["data_file"])
    t_obs = df["day"].values.astype(float)
    observed = df["cumulative_confirmed"].values.astype(float)
    population = config["population"]
    i0 = observed[0]
    y0 = np.array([population - 2 * i0, i0, i0, 0.0])
    return t_obs, observed, y0, population


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


def fit_mcmc_joint_rho(t_obs, observed, y0, population, seed):
    y0_scaled = y0 / STATE_SCALE
    observed_scaled = observed / STATE_SCALE  # NOTE: no division by rho here

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
        log_rho = pm.TruncatedNormal(
            "log_rho", mu=LOG_RHO_PRIOR_MU, sigma=LOG_RHO_PRIOR_SIGMA,
            lower=np.log(1e-6), upper=0.0,  # rho in (1e-6, 1], a reporting
                                             # fraction cannot exceed 1
            initval=LOG_RHO_PRIOR_MU,
        )
        rho = pm.Deterministic("rho", pt.exp(log_rho))

        ode_solution = ode_model(y0=y0_scaled, theta=[beta, sigma_p, gamma])
        simulated_scaled = rho * (ode_solution[:, 2] + ode_solution[:, 3])

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        pm.Normal("obs", mu=simulated_scaled, sigma=sigma_obs, observed=observed_scaled)

        idata = pm.sample(
            draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, cores=1,
            target_accept=TARGET_ACCEPT, random_seed=seed, progressbar=True,
        )

    posterior_means = {
        "beta": float(idata.posterior["beta"].mean()),
        "sigma": float(idata.posterior["sigma_p"].mean()),
        "gamma": float(idata.posterior["gamma"].mean()),
        "rho": float(idata.posterior["rho"].mean()),
    }
    r_hat_values = {
        var: float(pm.stats.rhat(idata, var_names=[var])[var].values)
        for var in ["beta", "sigma_p", "gamma", "rho"]
    }
    return posterior_means, r_hat_values


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return pd.read_csv(CHECKPOINT_PATH)
    return pd.DataFrame(columns=[
        "country", "beta", "sigma", "gamma", "rho", "rhat_beta",
        "rhat_sigma", "rhat_gamma", "rhat_rho", "fit_duration_seconds",
        "completed_at",
    ])


def append_checkpoint(row_dict):
    df_row = pd.DataFrame([row_dict])
    write_header = not os.path.exists(CHECKPOINT_PATH)
    df_row.to_csv(CHECKPOINT_PATH, mode="a", header=write_header, index=False)


def plot_fit(country_key, display_name, t_obs, observed, y0, population, params):
    from seir_model import solve_seir
    beta, sigma, gamma, rho = (params["beta"], params["sigma"],
                                params["gamma"], params["rho"])
    trajectory = solve_seir((beta, sigma, gamma), t_obs, y0, population)
    simulated_observed = rho * (trajectory[2] + trajectory[3])

    valid_mask = observed >= 50
    relative_residuals = ((simulated_observed - observed) / np.maximum(observed, 10.0))[valid_mask]
    rmse_relative = float(np.sqrt(np.mean(relative_residuals ** 2)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_obs, observed, "o", markersize=3, color="steelblue",
             label="Observed (JHU CSSE)")
    ax.plot(t_obs, simulated_observed, "-", color="#8172b3", linewidth=2,
             label="MCMC (joint rho) posterior-mean trajectory")
    ax.set_xlabel("Days since window start")
    ax.set_ylabel("Cumulative confirmed cases")
    ax.set_title(
        f"{display_name}: MCMC fit (rho estimated jointly) to real first-wave data\n"
        f"beta={beta:.3f}, sigma={sigma:.3f}, gamma={gamma:.3f}, rho={rho:.4f}, "
        f"relative RMSE={rmse_relative:.2%}"
    )
    ax.legend()
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"real_data_fit_mcmc_jointrho_{country_key}.png")
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

        print(f"\nFitting {config['display_name']} (MCMC, joint rho) -- "
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
        fit_start = time.time()

        t_obs, observed, y0, population = load_country_data(country_key)

        try:
            params, r_hat_values = fit_mcmc_joint_rho(
                t_obs, observed, y0, population, seed=RANDOM_SEED
            )
            fit_duration = time.time() - fit_start

            row = {
                "country": country_key,
                "beta": params["beta"],
                "sigma": params["sigma"],
                "gamma": params["gamma"],
                "rho": params["rho"],
                "rhat_beta": r_hat_values["beta"],
                "rhat_sigma": r_hat_values["sigma_p"],
                "rhat_gamma": r_hat_values["gamma"],
                "rhat_rho": r_hat_values["rho"],
                "fit_duration_seconds": fit_duration,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_checkpoint(row)

            rmse = plot_fit(country_key, config["display_name"], t_obs, observed,
                             y0, population, params)

            print(f"DONE in {fit_duration:.0f}s -- beta={params['beta']:.4f} "
                  f"sigma={params['sigma']:.4f} gamma={params['gamma']:.4f} "
                  f"rho={params['rho']:.6f} relative RMSE={rmse:.2%} "
                  f"(r_hat: {r_hat_values['beta']:.3f}/{r_hat_values['sigma_p']:.3f}/"
                  f"{r_hat_values['gamma']:.3f}/{r_hat_values['rho']:.3f}) "
                  f"-- checkpoint saved.")
        except Exception as e:
            print(f"FAILED: {config['display_name']} -- {e}. "
                  f"Will be retried on the next run.")

    final_df = load_checkpoint()
    if len(final_df) > 0:
        final_df.to_csv(os.path.join(OUTPUT_DIR, "real_data_mcmc_jointrho_summary.csv"),
                         index=False)
        print("\nFinal summary:")
        print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
