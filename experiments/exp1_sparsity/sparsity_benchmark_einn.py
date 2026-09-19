"""
sparsity_benchmark_einn.py

Data-sparsity benchmark for SEIR parameter recovery -- physics-informed
neural network (epidemiology-informed neural network, EINN) estimator.

This script mirrors the experimental design of the companion classical
baseline (sparsity_benchmark_nls.py) so that the two methods are directly
comparable: the same synthetic ground-truth outbreak, the same five
data-sparsity levels, and the same number of random-initialization repeats
per level.

IMPLEMENTATION NOTE (important for anyone adapting this script)
------------------------------------------------------------------
The training loop is implemented directly in PyTorch, without a
higher-level PINN framework. This is a deliberate choice, not a stylistic
one: an earlier version of this experiment built on a popular PINN library
exhibited a silent optimization failure in which the transmission-rate
parameter (beta) received an essentially zero gradient (order 1e-20)
throughout training, while the other two parameters trained normally. The
root cause, found through direct inspection of per-parameter gradients,
was an incorrect non-dimensionalization: when each compartment (S, E, I,
R) is rescaled by its own typical magnitude for numerical conditioning,
the ODE residual equations must be rescaled by the CORRESPONDING RATIO OF
COMPARTMENT SCALES, not simply divided by each compartment's own scale.
Omitting this ratio silently produces a residual that is dimensionally
inconsistent, which in this particular parameter regime happened to leave
beta's gradient near the floating-point noise floor while leaving the
other two parameters' gradients small but non-zero -- an easy failure mode
to miss without directly inspecting per-parameter gradients rather than
only the aggregate loss. See the residual definitions below for the
corrected, dimensionally-consistent form. This is documented here in
detail because it is a genuine pitfall for anyone implementing an inverse
problem over a system with compartments of very different magnitude, and
because it did not reveal itself as a training divergence or NaN, but as
one specific parameter appearing to converge to a stable, plausible-looking
(but data-independent) value.

1. Generates the same synthetic ground-truth SEIR outbreak used by the
   classical baseline (no observation noise; noise robustness is addressed
   in a separate companion experiment).
2. For each data-sparsity level (15%, 30%, 45%, 60%, 100% of the full
   epidemic cycle), trains a small feed-forward neural network that maps
   time -> (S, E, I, R), constrained by:
      (a) a data-fitting loss against the observed infectious curve I(t)
          over the available (sparse) time window, and
      (b) a physics loss that penalizes violation of the (correctly
          non-dimensionalized) SEIR ODE system, evaluated via automatic
          differentiation, over the FULL time window -- this is what
          allows the network to extrapolate beyond the observed data
          using the epidemiological constraints alone.
   The initial condition is enforced exactly via a hard constraint in the
   network's output layer (rather than as a soft loss term), and the
   transmission rate (beta), incubation rate (sigma), and recovery rate
   (gamma) are treated as trainable scalars, jointly optimized with the
   network weights (inverse-problem formulation).
3. Repeats each fit from multiple random initializations (seeds) to assess
   training robustness, exactly as done for the classical baseline.
4. Reports, for each sparsity level: the median and interquartile range of
   the relative parameter-recovery error.
5. Saves raw per-run results, a summary table, and a summary plot in the
   same format as the classical baseline, to support direct comparison.


Outputs are written to ./results/ (created automatically). A GPU is not
required but will speed up training if available and detected
automatically.

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
torch.set_default_dtype(torch.float64)  # see implementation note above

POPULATION_SIZE = 1_000_000
BETA_TRUE = 0.35
SIGMA_TRUE = 1 / 5.2
GAMMA_TRUE = 1 / 10.0
PARAM_NAMES = ["beta", "sigma", "gamma"]
PARAMS_TRUE = np.array([BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE])

E0, I0, R0 = 1, 1, 0
S0 = POPULATION_SIZE - E0 - I0 - R0
Y0 = np.array([S0, E0, I0, R0], dtype=float)

DATA_FRACTIONS = [0.15, 0.30, 0.45, 0.60, 1.00]
N_SEEDS = 15
OBSERVATION_INTERVAL_DAYS = 1.0

N_COLLOCATION_POINTS = 1000
N_WARMUP_ITERATIONS = 1500     # phase 0: fit network to the initial-guess
                                # SEIR curve, so it starts from a realistic
                                # epidemic shape rather than a near-flat
                                # random function (see implementation note)
N_ADAM_ITERATIONS = 4000       # phase 1a: joint physics + data training
N_LBFGS_ITERATIONS = 500       # phase 1b: second-order refinement
LEARNING_RATE = 1e-3
NETWORK_HIDDEN_LAYERS = [32, 32, 32]

# Parameter bounds used both to sample random initial guesses and to keep
# the classical baseline and the EINN on comparable footing.
PARAM_BOUNDS_LOWER = np.array([0.01, 1 / 14, 1 / 21])
PARAM_BOUNDS_UPPER = np.array([2.00, 1 / 2, 1 / 3])

RANDOM_SEED_BASE = 12345


# ---------------------------------------------------------------------------
# Synthetic ground truth (identical procedure to the classical baseline)
# ---------------------------------------------------------------------------

def generate_full_trajectory():
    full_duration = determine_outbreak_duration(PARAMS_TRUE, Y0, POPULATION_SIZE)
    n_points = int(np.ceil(full_duration / OBSERVATION_INTERVAL_DAYS)) + 1
    time_grid = np.linspace(0, full_duration, n_points)
    trajectory = solve_seir(PARAMS_TRUE, time_grid, Y0, POPULATION_SIZE)
    return time_grid, trajectory, full_duration


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
    """Wrap the raw network so its output exactly satisfies the initial
    condition at t=0, via a hard constraint: y(t) = y0 + (1 - exp(-t)) * NN(t).
    This avoids relying on a soft initial-condition loss term, which a
    randomly-initialized network only approximately satisfies.
    """
    def forward(t):
        raw = net(t)
        factor = 1 - torch.exp(-t)
        return y0_scaled + factor * raw
    return forward


# ---------------------------------------------------------------------------
# EINN training for a single (fraction, seed) configuration
# ---------------------------------------------------------------------------

def train_einn(t_obs, i_obs, full_duration, y0, population, compartment_scale,
                seed):
    """Train an EINN to recover (beta, sigma, gamma) from a sparse
    observation window of the infectious curve, using the full,
    correctly non-dimensionalized SEIR ODE system as a physics constraint
    over [0, full_duration].

    Returns
    -------
    fitted_params : numpy.ndarray, [beta, sigma, gamma]
    final_loss : float
    """
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

    # ---- Phase 0: warm-start the network to the SEIR curve implied by the
    # initial parameter guess (not the real data). This gives the network a
    # realistic epidemic shape before physics+data training begins, which
    # is necessary (see implementation note) for the gradient of beta to be
    # informative once phase 1 starts.
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

    # ---- Phase 1: joint training on the real (sparse) observed data and
    # the physics constraint, with beta/sigma/gamma now trainable.
    collocation_t = torch.linspace(
        0, full_duration, N_COLLOCATION_POINTS, device=DEVICE
    ).reshape(-1, 1).requires_grad_(True)

    t_obs_tensor = torch.tensor(t_obs.reshape(-1, 1), device=DEVICE)
    i_obs_scaled = torch.tensor((i_obs / scale_I).reshape(-1, 1), device=DEVICE)

    optimizer = torch.optim.Adam(
        list(net.parameters()) + [log_beta, log_sigma, log_gamma],
        lr=LEARNING_RATE,
    )

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

        # Correctly non-dimensionalized residuals: each equation is
        # rescaled by the RATIO of compartment scales implied by dividing
        # the original (real-unit) ODE by the scale of its own compartment
        # -- not simply divided by that compartment's own scale, which
        # silently starves beta's gradient (see implementation note above).
        r_s = ds + beta * s * i * (scale_I / population)
        r_e = de - beta * s * i * (scale_S * scale_I) / (population * scale_E) \
            + sigma * e
        r_i = di - sigma * e * (scale_E / scale_I) + gamma * i
        r_r = dr - gamma * i * (scale_I / scale_R)

        physics_loss = (r_s ** 2).mean() + (r_e ** 2).mean() \
            + (r_i ** 2).mean() + (r_r ** 2).mean()

        y_data = forward(t_obs_tensor)
        data_loss = ((y_data[:, 2:3] - i_obs_scaled) ** 2).mean()

        return physics_loss + 100 * data_loss

    # ---- Phase 1a: Adam, for broad exploration of the loss landscape.
    for _ in range(N_ADAM_ITERATIONS):
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()

    # ---- Phase 1b: L-BFGS refinement. Standard practice in PINN training:
    # Adam explores broadly, L-BFGS uses second-order (curvature)
    # information to converge much further within the basin Adam found.
    # Note: this consistently reduces the final loss substantially (an
    # order of magnitude or more, in development testing), but does not
    # by itself resolve genuine parameter non-identifiability -- if
    # several (beta, sigma, gamma, network) combinations fit the data and
    # physics almost equally well (see the companion FIM identifiability
    # analysis), L-BFGS will converge faithfully to whichever such
    # solution is nearest the point Adam left off at, not necessarily the
    # true parameter values. This is an expected, reportable finding, not
    # a training deficiency to be tuned away.
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

    for fraction in DATA_FRACTIONS:
        n_obs = max(int(np.floor(fraction * len(time_grid))), 5)
        t_obs = time_grid[:n_obs]
        i_obs = infectious_curve[:n_obs]

        for seed_idx in range(N_SEEDS):
            fitted_params, final_loss = train_einn(
                t_obs, i_obs, full_duration, Y0, POPULATION_SIZE,
                compartment_scale, seed=RANDOM_SEED_BASE + seed_idx,
            )
            relative_error = np.abs(fitted_params - PARAMS_TRUE) / PARAMS_TRUE
            raw_results.append({
                "fraction": fraction,
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

        print(f"Fraction {fraction:.0%}: {n_obs} observations, "
              f"{N_SEEDS} EINN fits completed.")

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "einn_raw_results.csv"), index=False)

    summary_rows = []
    for fraction in DATA_FRACTIONS:
        subset = raw_df[raw_df["fraction"] == fraction]
        summary_rows.append({
            "fraction": fraction,
            "n_seeds": len(subset),
            "median_mean_rel_error": subset["mean_rel_error"].median(),
            "q25_mean_rel_error": subset["mean_rel_error"].quantile(0.25),
            "q75_mean_rel_error": subset["mean_rel_error"].quantile(0.75),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "einn_summary.csv"), index=False)

    print("\nSummary (EINN):")
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["fraction"] * 100, summary_df["median_mean_rel_error"],
            marker="o", color="darkorange", label="Median relative error (EINN)")
    ax.fill_between(
        summary_df["fraction"] * 100,
        summary_df["q25_mean_rel_error"],
        summary_df["q75_mean_rel_error"],
        alpha=0.2, color="darkorange", label="Interquartile range",
    )
    ax.set_xlabel("Fraction of the epidemic curve used for fitting (%)")
    ax.set_ylabel("Mean relative parameter-recovery error")
    ax.set_title("SEIR parameter recovery under data sparsity -- EINN")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "einn_sparsity_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
