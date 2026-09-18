"""
noise_benchmark_einn.py

Observation-noise robustness benchmark for SEIR parameter recovery --
physics-informed neural network (epidemiology-informed neural network,
EINN) estimator.

This script mirrors the experimental design of the companion classical
baseline (noise_benchmark_nls.py): the same synthetic ground-truth
outbreak, the same four noise levels, and the same number of random
noise-realization / initialization repeats per level. It reuses the same
training procedure (network architecture, warm-start, Adam + L-BFGS
schedule, and non-dimensionalization) as the companion data-sparsity EINN
script -- see that script's implementation note for a detailed account of
a non-dimensionalization bug found and fixed during development, which
applies identically here.

WHAT THIS SCRIPT DOES
----------------------
1. Generates the same synthetic ground-truth SEIR outbreak used by the
   classical baseline, observed over the FULL epidemic cycle (this
   experiment isolates the effect of observation noise from the effect of
   data sparsity, which is characterized separately).
2. For each noise level (0%, 5%, 15%, 30% relative Gaussian noise added to
   the observed infectious curve), trains an EINN -- with a hard
   initial-condition constraint, a warm-start phase, and joint physics +
   data training (Adam followed by L-BFGS refinement) -- to recover
   (beta, sigma, gamma) from the noisy observations.
3. Repeats each fit across multiple random noise realizations AND network
   initializations (one combined seed controls both, for reproducibility).
4. Reports, for each noise level: the median and interquartile range of
   the relative parameter-recovery error.
5. Saves raw per-run results, a summary table, and a summary plot, in the
   same format as the classical baseline, to support direct comparison.

HOW TO RUN
-----------
    pip install -r requirements.txt
    python noise_benchmark_einn.py

Outputs are written to ./results/ (created automatically). A GPU is not
required but will speed up training if available and detected
automatically.

DEPENDENCIES
-------------
numpy, scipy, matplotlib, pandas, torch, and the companion module
seir_model.py (must be in the same directory or on the Python path).

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
import torch

from seir_model import determine_outbreak_duration, solve_seir

# ---------------------------------------------------------------------------
# Configuration
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

# See the companion classical-baseline script for the rationale behind
# extending this range up to 100% (a possible NLS/EINN crossover was
# suggested by small-sample testing in the 30-70% range during development).
NOISE_LEVELS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 1.00]
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
# Synthetic ground truth and noise model (identical to the classical
# baseline, so the same noisy datasets are used by both methods when run
# with the same seeds)
# ---------------------------------------------------------------------------

def generate_full_trajectory():
    full_duration = determine_outbreak_duration(PARAMS_TRUE, Y0, POPULATION_SIZE)
    n_points = int(np.ceil(full_duration / OBSERVATION_INTERVAL_DAYS)) + 1
    time_grid = np.linspace(0, full_duration, n_points)
    trajectory = solve_seir(PARAMS_TRUE, time_grid, Y0, POPULATION_SIZE)
    return time_grid, trajectory, full_duration


def add_noise(i_true, noise_level, rng):
    if noise_level == 0.0:
        return i_true.copy()
    noise_std = np.maximum(noise_level * np.abs(i_true), NOISE_FLOOR_CASES)
    noisy = i_true + rng.normal(0.0, noise_std)
    return np.clip(noisy, 0.0, None)


# ---------------------------------------------------------------------------
# Network definition with a hard initial-condition constraint (identical
# to the companion sparsity-benchmark EINN script)
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
# EINN training for a single (noise_level, seed) configuration
# ---------------------------------------------------------------------------

def train_einn(t_obs, i_obs, full_duration, y0, population, compartment_scale,
                seed):
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

    # Phase 0: warm-start to the initial-guess SEIR curve.
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

    # Phase 1: joint training on the (noisy) observed data and physics.
    collocation_t = torch.linspace(
        0, full_duration, N_COLLOCATION_POINTS, device=DEVICE
    ).reshape(-1, 1).requires_grad_(True)

    t_obs_tensor = torch.tensor(t_obs.reshape(-1, 1), device=DEVICE)
    i_obs_scaled = torch.tensor((i_obs / scale_I).reshape(-1, 1), device=DEVICE)

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

        # Correctly non-dimensionalized residuals -- see the implementation
        # note in the companion sparsity-benchmark EINN script for why the
        # scale-ratio factors below are required, not optional.
        r_s = ds + beta * s * i * (scale_I / N)
        r_e = de - beta * s * i * (scale_S * scale_I) / (N * scale_E) \
            + sigma * e
        r_i = di - sigma * e * (scale_E / scale_I) + gamma * i
        r_r = dr - gamma * i * (scale_I / scale_R)

        physics_loss = (r_s ** 2).mean() + (r_e ** 2).mean() \
            + (r_i ** 2).mean() + (r_r ** 2).mean()

        y_data = forward(t_obs_tensor)
        data_loss = ((y_data[:, 2:3] - i_obs_scaled) ** 2).mean()

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
    print(f"Compartment scale used for loss normalization "
          f"[S, E, I, R]: {compartment_scale}")

    raw_results = []

    for noise_level in NOISE_LEVELS:
        for seed_idx in range(N_SEEDS):
            rng = np.random.default_rng(
                RANDOM_SEED_BASE + seed_idx + int(noise_level * 10000)
            )
            i_obs = add_noise(infectious_curve, noise_level, rng)

            fitted_params, final_loss = train_einn(
                time_grid, i_obs, full_duration, Y0, POPULATION_SIZE,
                compartment_scale, seed=RANDOM_SEED_BASE + seed_idx,
            )
            relative_error = np.abs(fitted_params - PARAMS_TRUE) / PARAMS_TRUE
            raw_results.append({
                "noise_level": noise_level,
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

        print(f"Noise level {noise_level:.0%}: {N_SEEDS} EINN fits completed.")

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "einn_noise_raw_results.csv"), index=False)

    summary_rows = []
    for noise_level in NOISE_LEVELS:
        subset = raw_df[raw_df["noise_level"] == noise_level]
        summary_rows.append({
            "noise_level": noise_level,
            "n_seeds": len(subset),
            "median_mean_rel_error": subset["mean_rel_error"].median(),
            "q25_mean_rel_error": subset["mean_rel_error"].quantile(0.25),
            "q75_mean_rel_error": subset["mean_rel_error"].quantile(0.75),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "einn_noise_summary.csv"), index=False)

    print("\nSummary (EINN, noise robustness):")
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["noise_level"] * 100, summary_df["median_mean_rel_error"],
            marker="o", color="darkorange", label="Median relative error (EINN)")
    ax.fill_between(
        summary_df["noise_level"] * 100,
        summary_df["q25_mean_rel_error"],
        summary_df["q75_mean_rel_error"],
        alpha=0.2, color="darkorange", label="Interquartile range",
    )
    ax.set_xlabel("Relative observation noise level (%)")
    ax.set_ylabel("Mean relative parameter-recovery error")
    ax.set_title("SEIR parameter recovery under observation noise -- EINN")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "einn_noise_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
