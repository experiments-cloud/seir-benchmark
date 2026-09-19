"""
real_data_fit_nls.py

Real-world validation of the SEIR parameter-recovery pipeline: fits
(beta, sigma, gamma, rho), transmission rate, incubation rate, recovery
rate, and reporting fraction, to real first-wave COVID-19 cumulative case
data for Italy and South Korea, via nonlinear least squares (the method
that performed most consistently throughout the synthetic-data benchmark
in this study).

Modeling assumptions, worth reading before interpreting results:

1. Observed cumulative confirmed cases are modeled as
   observed(t) = rho * (I(t) + R(t)), a noisy, underreported proxy
   for the cumulative number of individuals who have ever left the
   Exposed compartment. rho is fit jointly with the epidemiological
   parameters, not fixed.
2. The model's initial condition is anchored to the first observed data
   point of each country's window: E(0) = I(0) = (first observed count),
   R(0) = 0, S(0) = population - E(0) - I(0). This is a simplification,
   since the true early E/I split is unknown, but keeps day 0 of the
   model aligned with day 0 of the real data without introducing an
   additional free time-shift parameter.
3. Population sizes are fixed at approximate 2020 figures: Italy ~60.3
   million, South Korea ~51.8 million. A single well-mixed, homogeneous
   population is assumed, with no age structure, no spatial structure,
   and no changes in contact patterns over time such as lockdowns, so
   beta is treated as constant over the whole window, a known
   simplification particularly for Italy, whose national lockdown
   substantially reduced transmission partway through this window; this
   is discussed explicitly as a limitation in the manuscript, and is one
   plausible explanation if the fit to Italy is visibly worse than to
   South Korea.

Outputs are written to ./results/ (created automatically): fitted
parameters, fit-quality plots (data vs. model) for both countries, and a
summary table.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from seir_model import solve_seir

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COUNTRIES = {
    "italy": {"population": 60_300_000, "data_file": "real_data_italy.csv",
              "display_name": "Italy"},
    "korea": {"population": 51_780_000, "data_file": "real_data_korea.csv",
              "display_name": "South Korea"},
}

# Parameter bounds: beta, sigma, gamma, rho. rho is bounded strictly
# within (0, 1] since it represents a reporting fraction.
PARAM_BOUNDS_LOWER = np.array([0.01, 1 / 14, 1 / 21, 1e-4])
PARAM_BOUNDS_UPPER = np.array([2.00, 1 / 2, 1 / 3, 1.0])

N_RANDOM_RESTARTS = 20
RANDOM_SEED_BASE = 12345


def load_country_data(country_key):
    config = COUNTRIES[country_key]
    df = pd.read_csv(config["data_file"])
    t_obs = df["day"].values.astype(float)
    observed = df["cumulative_confirmed"].values.astype(float)
    population = config["population"]

    i0 = observed[0]
    y0 = np.array([population - 2 * i0, i0, i0, 0.0])
    return t_obs, observed, y0, population


def residuals(log_params, t_obs, observed, y0, population):
    beta, sigma, gamma, rho = np.exp(log_params)
    trajectory = solve_seir(
        (beta, sigma, gamma), t_obs, y0, population, rtol=1e-6, atol=1e-8
    )
    simulated_observed = rho * (trajectory[2] + trajectory[3])
    # Fit on a relative (percentage) scale so that the large dynamic range
    # of cumulative case counts (from single digits to hundreds of
    # thousands) does not cause the objective to be dominated entirely by
    # the largest values.
    return (simulated_observed - observed) / np.maximum(observed, 10.0)


def fit_country(country_key, rng):
    t_obs, observed, y0, population = load_country_data(country_key)

    log_lower = np.log(PARAM_BOUNDS_LOWER)
    log_upper = np.log(PARAM_BOUNDS_UPPER)

    best_result = None
    best_cost = np.inf
    for _ in range(N_RANDOM_RESTARTS):
        log_initial = rng.uniform(log_lower, log_upper)
        result = least_squares(
            residuals, log_initial, bounds=(log_lower, log_upper),
            args=(t_obs, observed, y0, population),
            method="trf", max_nfev=2000,
        )
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result

    fitted_params = np.exp(best_result.x)
    trajectory = solve_seir(
        (fitted_params[0], fitted_params[1], fitted_params[2]), t_obs, y0, population
    )
    simulated_observed = fitted_params[3] * (trajectory[2] + trajectory[3])

    # RMSE is computed only on days with at least 50 cumulative cases.
    # Earlier days have case counts of 1-4, where even a small absolute
    # deviation translates into an enormous percentage error (e.g., a
    # simulated value of 50 against an observed value of 1 is a 4,900%
    # relative error) -- this would let a handful of near-zero early days
    # dominate the metric regardless of how well the model tracks the
    # actual epidemic curve. This threshold was chosen after noticing
    # exactly this distortion during development (see the manuscript's
    # discussion of this choice).
    valid_mask = observed >= 50
    relative_residuals = ((simulated_observed - observed) / np.maximum(observed, 10.0))[valid_mask]
    rmse_relative = float(np.sqrt(np.mean(relative_residuals ** 2)))

    return {
        "country": country_key,
        "beta": fitted_params[0],
        "sigma": fitted_params[1],
        "gamma": fitted_params[2],
        "rho": fitted_params[3],
        "rmse_relative": rmse_relative,
        "t_obs": t_obs,
        "observed": observed,
        "simulated": simulated_observed,
    }


def plot_fit(fit_result, display_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fit_result["t_obs"], fit_result["observed"], "o", markersize=3,
             color="steelblue", label="Observed (JHU CSSE)")
    ax.plot(fit_result["t_obs"], fit_result["simulated"], "-", color="darkorange",
             linewidth=2, label="Fitted SEIR model")
    ax.set_xlabel("Days since window start")
    ax.set_ylabel("Cumulative confirmed cases")
    ax.set_title(
        f"{display_name}: NLS fit to real first-wave data\n"
        f"beta={fit_result['beta']:.3f}, sigma={fit_result['sigma']:.3f}, "
        f"gamma={fit_result['gamma']:.3f}, rho={fit_result['rho']:.4f}, "
        f"relative RMSE={fit_result['rmse_relative']:.2%}"
    )
    ax.legend()
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"real_data_fit_{fit_result['country']}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    rng = np.random.default_rng(RANDOM_SEED_BASE)
    summary_rows = []

    for country_key, config in COUNTRIES.items():
        print(f"\nFitting {config['display_name']} "
              f"({N_RANDOM_RESTARTS} random restarts)...")
        fit_result = fit_country(country_key, rng)
        plot_fit(fit_result, config["display_name"])

        print(f"  beta={fit_result['beta']:.4f}  sigma={fit_result['sigma']:.4f}  "
              f"gamma={fit_result['gamma']:.4f}  rho={fit_result['rho']:.4f}  "
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
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "real_data_nls_summary.csv"),
                       index=False)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
