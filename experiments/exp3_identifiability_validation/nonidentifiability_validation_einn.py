"""
nonidentifiability_validation_einn.py

Validation of the practical-identifiability diagnostic (companion script:
seir_identifiability_analysis.py) against actual parameter-recovery
behavior -- physics-informed neural network (EINN) estimator.

This script mirrors nonidentifiability_validation_nls.py exactly (same two
underreporting scenarios, same data volume, same noise level, same
seeds), using the EINN training procedure (hard initial-condition
constraint, warm-start, Adam + L-BFGS) from the companion sparsity/noise
benchmark scripts. See nonidentifiability_validation_nls.py's header for
the rationale behind the specific rho values chosen, including the
important nuance about why the FIM condition-number threshold alone does
not linearly predict empirical degradation.

The purpose of running both a classical and a physics-informed estimator
through the same non-identifiability test is to check whether the
structural limitation revealed by the FIM analysis affects both methods
equally (as expected, since it is a property of the model and data, not
of the fitting algorithm) or whether one method is more resilient to it
in practice.


Outputs are written to ./results/ (created automatically).

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from seir_model import determine_outbreak_duration, solve_seir

# ---------------------------------------------------------------------------
# Configuration (kept identical to the classical-baseline script wherever
# it affects comparability)
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

POPULATION_SIZE = 1_000_000
BETA_TRUE = 0.35
SIGMA_TRUE = 1 / 5.2
GAMMA_TRUE = 1 / 10.0
PARAM_NAMES = ["beta", "sigma", "gamma"]
PARAMS_TRUE = np.array([BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE])

E0, I0, R0 = 1, 1, 0
S0 = POPULATION_SIZE - E0 - I0 - R0
Y0 = np.array([S0, E0, I0, R0], dtype=float)

# See nonidentifiability_validation_nls.py for the rationale behind these
# specific rho values.
SCENARIOS = {
    "control_identifiable_rho030": 0.30,
    "test_nonidentifiable_rho1e-5": 0.00001,
}

NOISE_LEVEL = 0.10
NOISE_FLOOR_CASES = 1.0

N_SEEDS = 15
OBSERVATION_INTERVAL_DAYS = 1.0

N_COLLOCATION_POINTS = 1000
N_WARMUP_ITERATIONS = 1500
N_ADAM_ITERATIONS = 4000
N_LBFGS_ITERATIONS = 500
LEARNING_RATE = 1e-3
NETWORK_HIDDEN_LAYERS = [32, 32, 32]

PARAM_BOUNDS_LOWER = np.array([0.01, 1 / 14, 1 / 21])
PARAM_BOUNDS_UPPER = np.array([2.00, 1 / 2, 1 / 3])

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


def add_noise(observed_true, noise_level, rng):
    noise_std = np.maximum(noise_level * np.abs(observed_true), NOISE_FLOOR_CASES)
    noisy = observed_true + rng.normal(0.0, noise_std)
    return np.clip(noisy, 0.0, None)


# ---------------------------------------------------------------------------
# Network definition with a hard initial-condition constraint
# ---------------------------------------------------------------------------

def build_network():
    layers = []
    sizes = [1] + NETWORK_HIDDEN_LAYERS + [4]
    for i in range(len(sizes) - 2):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(torch.nn.Tanh())
    layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
    return torch.nn.Sequential(*layers).to(DEVICE)


def make_forward(net, y0_scaled):
    def forward(t):
        raw = net(t)
        factor = 1 - torch.exp(-t)
        return y0_scaled + factor * raw
    return forward


# ---------------------------------------------------------------------------
# EINN training under a given underreporting fraction (rho)
# ---------------------------------------------------------------------------

def train_einn(t_obs, observed, full_duration, y0, population,
                compartment_scale, rho, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(RANDOM_SEED_BASE + seed)

    log_params_true_scale = rng.uniform(
        np.log(PARAM_BOUNDS_LOWER), np.log(PARAM_BOUNDS_UPPER)
    )
    beta_init, sigma_init, gamma_init = np.exp(log_params_true_scale)

    log_beta = torch.tensor(np.log(beta_init), requires_grad=True, device=DEVICE)
    log_sigma = torch.tensor(np.log(sigma_init), requires_grad=True, device=DEVICE)
    log_gamma = torch.tensor(np.log(gamma_init), requires_grad=True, device=DEVICE)

    scale_S, scale_E, scale_I, scale_R = compartment_scale
    y0_scaled = torch.tensor(y0 / compartment_scale, device=DEVICE)

    net = build_network()
    forward = make_forward(net, y0_scaled)

    time_grid_warm = np.linspace(0, full_duration, 300)
    warm_trajectory = solve_seir(
        (beta_init, sigma_init, gamma_init), time_grid_warm, y0, population
    )
    warm_target = torch.tensor(
        (warm_trajectory / compartment_scale.reshape(-1, 1)).T, device=DEVICE
    )
    t_warm = torch.tensor(time_grid_warm.reshape(-1, 1), device=DEVICE)

    warmup_optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    for _ in range(N_WARMUP_ITERATIONS):
        warmup_optimizer.zero_grad()
        pred = forward(t_warm)
        warmup_loss = ((pred - warm_target) ** 2).mean()
        warmup_loss.backward()
        warmup_optimizer.step()

    collocation_t = torch.linspace(
        0, full_duration, N_COLLOCATION_POINTS, device=DEVICE
    ).reshape(-1, 1).requires_grad_(True)

    t_obs_tensor = torch.tensor(t_obs.reshape(-1, 1), device=DEVICE)
    # The observation is rho * I(t); the network's I-output is compared
    # against observed/rho (equivalently, the network's scaled I output
    # times rho is compared against observed/scale_I) -- here we rescale
    # the network's I output by rho before comparing to the noisy
    # underreported observation, keeping the physics residual (which
    # governs the true, unscaled I) unaffected by rho.
    observed_scaled = torch.tensor(
        (observed / (rho * scale_I)).reshape(-1, 1), device=DEVICE
    )

    N = population

    def compute_loss():
        y_col = forward(collocation_t)
        s, e, i, r = y_col[:, 0:1], y_col[:, 1:2], y_col[:, 2:3], y_col[:, 3:4]
        grad_outputs = torch.ones_like(s)
        ds = torch.autograd.grad(s, collocation_t, grad_outputs=grad_outputs,
                                  create_graph=True)[0]
        de = torch.autograd.grad(e, collocation_t, grad_outputs=grad_outputs,
                                  create_graph=True)[0]
        di = torch.autograd.grad(i, collocation_t, grad_outputs=grad_outputs,
                                  create_graph=True)[0]
        dr = torch.autograd.grad(r, collocation_t, grad_outputs=grad_outputs,
                                  create_graph=True)[0]

        beta = torch.exp(log_beta)
        sigma = torch.exp(log_sigma)
        gamma = torch.exp(log_gamma)

        r_s = ds + beta * s * i * (scale_I / N)
        r_e = de - beta * s * i * (scale_S * scale_I) / (N * scale_E) \
            + sigma * e
        r_i = di - sigma * e * (scale_E / scale_I) + gamma * i
        r_r = dr - gamma * i * (scale_I / scale_R)

        physics_loss = (r_s ** 2).mean() + (r_e ** 2).mean() \
            + (r_i ** 2).mean() + (r_r ** 2).mean()

        y_data = forward(t_obs_tensor)
        # y_data[:,2:3] is the network's I output (scaled); the observed
        # underreported signal, once divided by rho and scale_I, lives on
        # the same scale as the network's raw I output.
        data_loss = ((y_data[:, 2:3] - observed_scaled) ** 2).mean()

        return physics_loss + 100 * data_loss

    optimizer = torch.optim.Adam(
        list(net.parameters()) + [log_beta, log_sigma, log_gamma],
        lr=LEARNING_RATE,
    )
    for _ in range(N_ADAM_ITERATIONS):
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()

    lbfgs_optimizer = torch.optim.LBFGS(
        list(net.parameters()) + [log_beta, log_sigma, log_gamma],
        lr=1.0, max_iter=N_LBFGS_ITERATIONS, line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs_optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        return loss

    lbfgs_optimizer.step(closure)
    final_loss = compute_loss().item()

    fitted_params = np.array([
        float(torch.exp(log_beta.detach())),
        float(torch.exp(log_sigma.detach())),
        float(torch.exp(log_gamma.detach())),
    ])
    return fitted_params, final_loss


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def main():
    print(f"Using device: {DEVICE}")
    time_grid, full_trajectory, full_duration = generate_full_trajectory()
    infectious_curve = full_trajectory[2]

    print(f"Full outbreak duration: {full_duration:.1f} days "
          f"({len(time_grid)} daily observations).")

    compartment_scale = np.maximum(np.max(np.abs(full_trajectory), axis=1), 1.0)

    raw_results = []

    for scenario_name, rho in SCENARIOS.items():
        true_observed = rho * infectious_curve

        for seed_idx in range(N_SEEDS):
            rng = np.random.default_rng(RANDOM_SEED_BASE + seed_idx)
            observed_noisy = add_noise(true_observed, NOISE_LEVEL, rng)

            fitted_params, final_loss = train_einn(
                time_grid, observed_noisy, full_duration, Y0, POPULATION_SIZE,
                compartment_scale, rho, seed=RANDOM_SEED_BASE + seed_idx,
            )
            relative_error = np.abs(fitted_params - PARAMS_TRUE) / PARAMS_TRUE
            raw_results.append({
                "scenario": scenario_name,
                "rho": rho,
                "seed": seed_idx,
                "final_loss": final_loss,
                "beta_fitted": fitted_params[0],
                "sigma_fitted": fitted_params[1],
                "gamma_fitted": fitted_params[2],
                "beta_rel_error": relative_error[0],
                "sigma_rel_error": relative_error[1],
                "gamma_rel_error": relative_error[2],
                "mean_rel_error": float(np.mean(relative_error)),
            })

        print(f"Scenario {scenario_name} (rho={rho}): {N_SEEDS} EINN fits completed.")

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "nonid_einn_raw_results.csv"), index=False)

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
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "nonid_einn_summary.csv"), index=False)

    print("\nSummary (EINN, identifiability validation):")
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(SCENARIOS))
    width = 0.25
    for i, param in enumerate(["beta", "sigma", "gamma"]):
        values = summary_df[f"median_{param}_rel_error"]
        ax.bar(x + i * width, values, width, label=param, color=["#4c72b0", "#dd8452", "#55a868"][i])
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{s}\n(rho={r})" for s, r in
                         zip(summary_df["scenario"], summary_df["rho"])])
    ax.set_ylabel("Median relative parameter-recovery error")
    ax.set_title("EINN: parameter recovery in predicted identifiable\n"
                  "vs. non-identifiable regimes (same data volume and noise)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "nonid_einn_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
