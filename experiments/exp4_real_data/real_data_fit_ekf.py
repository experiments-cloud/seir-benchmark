"""
real_data_fit_ekf.py

Real-world validation of the SEIR parameter-recovery pipeline, Extended
Kalman Filter (EKF) baseline. Fits (beta, sigma, gamma) to real first-wave
COVID-19 cumulative case data for Italy and South Korea.

See real_data_fit_nls.py for the shared modeling assumptions (observation
model, initial-condition handling, population sizes, and the important
caveat about constant-beta SEIR being misspecified for countries with an
abrupt lockdown, such as Italy).

Unlike real_data_fit_nls.py, which estimates the reporting fraction rho
jointly with (beta, sigma, gamma), this EKF implementation keeps rho
fixed, at the value already estimated by the NLS fit for the same
country (real_data_fit_nls.py must be run first). Extending the EKF's
augmented state to include rho as a fifth estimated quantity would
require re-deriving the filter's Jacobian for an 8-dimensional state; this
was judged not worth the added complexity for a single confirmatory
real-data experiment, given that the main comparative claims of this study
already rest on the synthetic-data benchmarks, which do not have this
limitation. This is a real asymmetry between the two methods on this
specific experiment and is stated plainly here and in the manuscript,
rather than left implicit.

Outputs are written to ./results/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seir_model import solve_seir
from ekf_seir import run_ekf

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COUNTRIES = {
    "italy": {"population": 60_300_000, "data_file": "real_data_italy.csv",
              "display_name": "Italy"},
    "korea": {"population": 51_780_000, "data_file": "real_data_korea.csv",
              "display_name": "South Korea"},
}

PARAM_INIT_GUESS = np.array([0.30, 0.20, 0.10])


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
            "real_data_nls_summary.csv not found -- run real_data_fit_nls.py "
            "first (this script reuses its rho estimate)."
        )
    nls_summary = pd.read_csv(nls_summary_path)
    row = nls_summary[nls_summary["country"] == config["display_name"]]
    if len(row) == 0:
        raise ValueError(f"No NLS result found for {config['display_name']} "
                          f"in real_data_nls_summary.csv.")
    rho = float(row["rho"].iloc[0])

    return t_obs, observed, y0, population, rho


def fit_country(country_key):
    t_obs, observed, y0, population, rho = load_country_data_and_rho(country_key)

    # Daily-resolution EKF requires a full day-by-day time grid, since it
    # is a sequential filter; the real data is already daily, so no
    # resampling is needed.
    obs_noise_std = np.maximum(0.10 * np.abs(observed), 5.0)

    state_estimates, fitted_params = run_ekf(
        observed, dt=1.0, y0=y0, param_init=PARAM_INIT_GUESS,
        population=population, rho=rho, observation_noise_std=obs_noise_std,
        observation_weights=np.array([0.0, 0.0, 1.0, 1.0]),  # observe I + R
    )

    simulated_observed = rho * (state_estimates[:, 2] + state_estimates[:, 3])
    # See real_data_fit_nls.py for the rationale behind excluding
    # near-zero early days from this metric.
    valid_mask = observed >= 50
    relative_residuals = ((simulated_observed - observed) / np.maximum(observed, 10.0))[valid_mask]
    rmse_relative = float(np.sqrt(np.mean(relative_residuals ** 2)))

    return {
        "country": country_key,
        "beta": fitted_params[0],
        "sigma": fitted_params[1],
        "gamma": fitted_params[2],
        "rho": rho,
        "rmse_relative": rmse_relative,
        "t_obs": t_obs,
        "observed": observed,
        "simulated": simulated_observed,
    }


def plot_fit(fit_result, display_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fit_result["t_obs"], fit_result["observed"], "o", markersize=3,
             color="steelblue", label="Observed (JHU CSSE)")
    ax.plot(fit_result["t_obs"], fit_result["simulated"], "-", color="seagreen",
             linewidth=2, label="EKF-filtered SEIR trajectory")
    ax.set_xlabel("Days since window start")
    ax.set_ylabel("Cumulative confirmed cases")
    ax.set_title(
        f"{display_name}: EKF fit to real first-wave data\n"
        f"beta={fit_result['beta']:.3f}, sigma={fit_result['sigma']:.3f}, "
        f"gamma={fit_result['gamma']:.3f}, rho={fit_result['rho']:.4f} (fixed, from NLS), "
        f"relative RMSE={fit_result['rmse_relative']:.2%}"
    )
    ax.legend()
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"real_data_fit_ekf_{fit_result['country']}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    summary_rows = []
    for country_key, config in COUNTRIES.items():
        print(f"\nFitting {config['display_name']} (EKF)...")
        fit_result = fit_country(country_key)
        plot_fit(fit_result, config["display_name"])

        print(f"  beta={fit_result['beta']:.4f}  sigma={fit_result['sigma']:.4f}  "
              f"gamma={fit_result['gamma']:.4f}  rho={fit_result['rho']:.4f} (fixed)  "
              f"relative RMSE={fit_result['rmse_relative']:.2%}")

        summary_rows.append({
            "country": config["display_name"],
            "beta": fit_result["beta"],
            "sigma": fit_result["sigma"],
            "gamma": fit_result["gamma"],
            "rho": fit_result["rho"],
            "implied_R0": fit_result["beta"] / fit_result["gamma"],
            "relative_rmse": fit_result["rmse_relative"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "real_data_ekf_summary.csv"),
                       index=False)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
