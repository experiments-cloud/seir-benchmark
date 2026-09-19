"""
seir_identifiability_analysis.py

Structural/practical identifiability diagnostic for the SEIR epidemic model
based on the Fisher Information Matrix (FIM).

1. Simulates the SEIR compartmental model (S, E, I, R) from a synthetic
   ground-truth parameter set using SciPy's ODE solver. The simulation
   horizon is determined DYNAMICALLY so that it always covers a full
   epidemic cycle (growth, peak, and decline of the infectious curve),
   rather than an arbitrary fixed number of days -- this matters because
   truncating the curve mid-growth discards information (particularly
   about the recovery rate, gamma) that is only available once the
   outbreak has passed its peak.
2. Computes the sensitivity of the observed outputs to each model parameter
   (beta: transmission rate, sigma: incubation rate, gamma: recovery rate)
   via centered finite differences.
3. Builds the Fisher Information Matrix under three observability scenarios:
      - "I_only":           only the infectious compartment is observed
      - "I_and_R":          infectious and recovered compartments observed
      - "I_underreported":  infectious compartment observed with a fixed
                             underreporting fraction
4. Diagnoses practical identifiability from the FIM eigenstructure
   (condition number, best/worst-determined parameter combinations).
5. Saves numerical results, a summary table, and a reference plot.

The SEIR inverse problem (recovering beta, sigma, gamma from observed case
data) is known to suffer from partial parameter confounding, particularly
between beta and sigma, when only the infectious curve is observed. Running
this diagnostic BEFORE fitting any model (classical or neural-network-based)
tells you which parameter combinations can, in principle, be recovered from
your data, and which cannot regardless of the fitting method used.


Outputs are written to ./results/ (created automatically).

"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference ("true") parameter values used to generate synthetic data.
# Adjust these to match a specific real-world outbreak of interest.
POPULATION_SIZE = 1_000_000
BETA_TRUE = 0.35            # transmission rate
SIGMA_TRUE = 1 / 5.2        # incubation rate (mean incubation period ~5.2 days)
GAMMA_TRUE = 1 / 10.0       # recovery rate (mean infectious period ~10 days)
UNDERREPORTING_FRACTION = 0.3  # used only in the "I_underreported" scenario

PARAM_NAMES = ["beta", "sigma", "gamma"]
PARAMS_TRUE = np.array([BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE])

# Initial conditions: one exposed, one infectious individual, rest susceptible
E0, I0, R0 = 1, 1, 0
S0 = POPULATION_SIZE - E0 - I0 - R0
Y0 = [S0, E0, I0, R0]

N_TIME_POINTS = 200

# Search horizon used only to locate the end of the outbreak; must be long
# enough to comfortably contain a full epidemic cycle for the parameter
# regime under study. Increase this if the outbreak does not resolve within
# it (the script will warn if that happens).
DURATION_SEARCH_HORIZON_DAYS = 730

# The outbreak is considered "resolved" once the infectious compartment
# drops below this fraction of its peak value, after having passed the peak.
DECLINE_FRACTION_OF_PEAK = 0.01

# A fixed buffer (in days) added after the detected end of the outbreak,
# so the observed curve visibly flattens out rather than ending right at
# the threshold crossing.
POST_OUTBREAK_BUFFER_DAYS = 15

# Assumed relative observation noise, used to scale the Fisher Information
# Matrix (does not add actual noise here -- see companion noise-robustness
# experiment for that).
RELATIVE_NOISE_STD = 0.10

OBSERVATION_SCENARIOS = ["I_only", "I_and_R", "I_underreported"]

# Condition-number threshold above which a parameter combination is flagged
# as practically non-identifiable. This is a conventional choice; sensitivity
# of conclusions to this threshold should be checked.
CONDITION_NUMBER_THRESHOLD = 1e4


# ---------------------------------------------------------------------------
# SEIR model
# ---------------------------------------------------------------------------

def seir_rhs(t, y, beta, sigma, gamma, population):
    """Right-hand side of the SEIR ODE system."""
    S, E, I, R = y
    dS = -beta * S * I / population
    dE = beta * S * I / population - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


def solve_seir(params, t_eval, y0=Y0, population=POPULATION_SIZE):
    """Integrate the SEIR system for a given parameter set over the
    provided time grid `t_eval`.

    Returns
    -------
    numpy.ndarray of shape (4, len(t_eval)): rows are S, E, I, R.
    """
    beta, sigma, gamma = params
    solution = solve_ivp(
        seir_rhs,
        [t_eval[0], t_eval[-1]],
        y0,
        t_eval=t_eval,
        args=(beta, sigma, gamma, population),
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return solution.y


def determine_outbreak_duration(
    params,
    y0=Y0,
    population=POPULATION_SIZE,
    search_horizon_days=DURATION_SEARCH_HORIZON_DAYS,
    decline_fraction=DECLINE_FRACTION_OF_PEAK,
    buffer_days=POST_OUTBREAK_BUFFER_DAYS,
):
    """Determine how many days are needed to observe a full epidemic cycle
    (growth, peak, and decline of the infectious compartment) for a given
    parameter set, instead of assuming a fixed, arbitrary horizon.

    Returns
    -------
    float: recommended simulation duration, in days.
    """
    search_grid = np.linspace(0, search_horizon_days, 2000)
    trajectory = solve_seir(params, t_eval=search_grid, y0=y0, population=population)
    infectious_curve = trajectory[2]

    peak_index = int(np.argmax(infectious_curve))
    peak_value = infectious_curve[peak_index]
    decline_threshold = decline_fraction * peak_value

    post_peak = infectious_curve[peak_index:]
    below_threshold = np.where(post_peak < decline_threshold)[0]

    if below_threshold.size == 0:
        print(
            "WARNING: the infectious curve did not fall below "
            f"{decline_fraction:.1%} of its peak within the "
            f"{search_horizon_days}-day search horizon. Consider increasing "
            "DURATION_SEARCH_HORIZON_DAYS. Falling back to the full search "
            "horizon."
        )
        return float(search_horizon_days)

    end_index = peak_index + below_threshold[0]
    end_day = search_grid[end_index] + buffer_days
    return float(min(end_day, search_horizon_days))


# ---------------------------------------------------------------------------
# Observation operators
# ---------------------------------------------------------------------------

def observe(y, scenario, underreporting_fraction=UNDERREPORTING_FRACTION):
    """Map the full SEIR state to the observed quantities for a given
    observability scenario.
    """
    S, E, I, R = y
    if scenario == "I_only":
        return I.reshape(1, -1)
    elif scenario == "I_and_R":
        return np.vstack([I, R])
    elif scenario == "I_underreported":
        return (underreporting_fraction * I).reshape(1, -1)
    else:
        raise ValueError(f"Unknown observation scenario: {scenario}")


# ---------------------------------------------------------------------------
# Parameter sensitivity (centered finite differences)
# ---------------------------------------------------------------------------

def compute_sensitivity(params, t_eval, scenario, relative_step=1e-4):
    """Compute the Jacobian of the observed outputs with respect to the
    model parameters using centered finite differences.

    Returns
    -------
    sensitivity : numpy.ndarray, shape (n_observations, n_parameters)
    base_observation : numpy.ndarray, the observation at `params`
    """
    n_params = len(params)
    base_observation = observe(solve_seir(params, t_eval), scenario)
    sensitivity = np.zeros((base_observation.size, n_params))

    for i in range(n_params):
        step = relative_step * max(abs(params[i]), 1e-8)

        params_plus = np.array(params, dtype=float)
        params_plus[i] += step
        obs_plus = observe(solve_seir(params_plus, t_eval), scenario)

        params_minus = np.array(params, dtype=float)
        params_minus[i] -= step
        obs_minus = observe(solve_seir(params_minus, t_eval), scenario)

        sensitivity[:, i] = (obs_plus.flatten() - obs_minus.flatten()) / (2 * step)

    return sensitivity, base_observation


# ---------------------------------------------------------------------------
# Fisher Information Matrix
# ---------------------------------------------------------------------------

def fisher_information_matrix(sensitivity, base_observation,
                               relative_noise_std=RELATIVE_NOISE_STD):
    """Build the Fisher Information Matrix under an assumed Gaussian,
    heteroscedastic observation noise model with standard deviation
    proportional to the magnitude of each observation (a common choice
    for epidemic case-count data).
    """
    observation_magnitude = np.abs(base_observation.flatten())
    noise_std = np.maximum(relative_noise_std * observation_magnitude, 1.0)
    weight_matrix = np.diag(1.0 / noise_std ** 2)
    return sensitivity.T @ weight_matrix @ sensitivity


# ---------------------------------------------------------------------------
# Identifiability diagnostic
# ---------------------------------------------------------------------------

def diagnose_identifiability(fim, param_names,
                              condition_threshold=CONDITION_NUMBER_THRESHOLD):
    """Diagnose practical identifiability from the eigenstructure of the FIM.

    A large condition number indicates that some linear combination of
    parameters is poorly constrained by the data (near-unidentifiable),
    even if the matrix is formally non-singular.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(fim)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalues_safe = np.where(eigenvalues > 1e-15, eigenvalues, 1e-15)
    condition_number = eigenvalues_safe[0] / eigenvalues_safe[-1]

    best_direction = eigenvectors[:, 0]
    worst_direction = eigenvectors[:, -1]

    return {
        "condition_number": float(condition_number),
        "eigenvalues": eigenvalues.tolist(),
        "practically_identifiable": bool(condition_number < condition_threshold),
        "best_determined_combination": {
            param_names[i]: float(best_direction[i]) for i in range(len(param_names))
        },
        "worst_determined_combination": {
            param_names[i]: float(worst_direction[i]) for i in range(len(param_names))
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    outbreak_duration_days = determine_outbreak_duration(PARAMS_TRUE)
    time_grid = np.linspace(0, outbreak_duration_days, N_TIME_POINTS)
    print(
        f"Simulation horizon set dynamically to "
        f"{outbreak_duration_days:.1f} days (full epidemic cycle "
        f"+ {POST_OUTBREAK_BUFFER_DAYS}-day buffer)."
    )

    all_reports = {}

    # Reference trajectory, saved for visual inspection
    reference_trajectory = solve_seir(PARAMS_TRUE, time_grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, compartment_label in enumerate(["S", "E", "I", "R"]):
        ax.plot(time_grid, reference_trajectory[i], label=compartment_label)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of individuals")
    ax.set_title("Reference SEIR trajectory (synthetic ground truth, full cycle)")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "reference_seir_trajectory.png"), dpi=150)
    plt.close(fig)

    for scenario in OBSERVATION_SCENARIOS:
        sensitivity, base_observation = compute_sensitivity(
            PARAMS_TRUE, time_grid, scenario
        )
        fim = fisher_information_matrix(sensitivity, base_observation)
        report = diagnose_identifiability(fim, PARAM_NAMES)
        all_reports[scenario] = report

        print(f"\n=== Scenario: {scenario} ===")
        print(f"FIM condition number: {report['condition_number']:.3e}")
        print(f"Practically identifiable (threshold 1e4): "
              f"{report['practically_identifiable']}")
        print(f"Worst-determined parameter combination: "
              f"{report['worst_determined_combination']}")

    all_reports["metadata"] = {
        "outbreak_duration_days": outbreak_duration_days,
        "n_time_points": N_TIME_POINTS,
    }

    with open(os.path.join(OUTPUT_DIR, "identifiability_report.json"), "w") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    summary_rows = [
        {
            "scenario": scenario,
            "condition_number": report["condition_number"],
            "practically_identifiable": report["practically_identifiable"],
        }
        for scenario, report in all_reports.items()
        if scenario != "metadata"
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "identifiability_summary.csv"),
                       index=False)

    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
