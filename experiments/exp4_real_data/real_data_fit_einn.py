"""
real_data_fit_einn.py

Real-world validation of the SEIR parameter-recovery pipeline --
epidemiology-informed neural network (EINN) estimator. Fits
(beta, sigma, gamma, rho) jointly to real first-wave COVID-19 cumulative
case data for Italy and South Korea, using the same training procedure
(hard initial-condition constraint, warm-start, Adam + L-BFGS, correctly
non-dimensionalized physics residuals) as the synthetic-data EINN
benchmark scripts.

See real_data_fit_nls.py for the shared modeling assumptions (observation
model, initial-condition handling, population sizes, and the caveat about
constant-beta SEIR being misspecified for countries with an abrupt
lockdown, such as Italy).


Outputs are written to ./results/.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from seir_model import solve_seir

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

COUNTRIES = {
    "italy": {"population": 60_300_000, "data_file": "real_data_italy.csv",
              "display_name": "Italy"},
    "korea": {"population": 51_780_000, "data_file": "real_data_korea.csv",
              "display_name": "South Korea"},
}

PARAM_BOUNDS_LOWER = np.array([0.01, 1 / 14, 1 / 21, 1e-4])
PARAM_BOUNDS_UPPER = np.array([2.00, 1 / 2, 1 / 3, 1.0])
PARAM_INIT_GUESS = np.array([0.30, 0.20, 0.10, 0.01])

N_COLLOCATION_POINTS = 1000
N_WARMUP_ITERATIONS = 1500
N_ADAM_ITERATIONS = 5000
N_LBFGS_ITERATIONS = 500
LEARNING_RATE = 1e-3
NETWORK_HIDDEN_LAYERS = [32, 32, 32]

RANDOM_SEED = 12345


def load_country_data(country_key):
    config = COUNTRIES[country_key]
    df = pd.read_csv(config["data_file"])
    t_obs = df["day"].values.astype(float)
    observed = df["cumulative_confirmed"].values.astype(float)
    population = config["population"]

    i0 = observed[0]
    y0 = np.array([population - 2 * i0, i0, i0, 0.0])
    return t_obs, observed, y0, population


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


def train_einn(t_obs, observed, y0, population):
    full_duration = float(t_obs[-1])
    compartment_scale = np.array([
        population, max(observed[-1], 1.0), max(observed[-1], 1.0), max(observed[-1], 1.0)
    ])
    scale_S, scale_E, scale_I, scale_R = compartment_scale

    torch.manual_seed(RANDOM_SEED)
    log_beta = torch.tensor(np.log(PARAM_INIT_GUESS[0]), requires_grad=True, device=DEVICE)
    log_sigma = torch.tensor(np.log(PARAM_INIT_GUESS[1]), requires_grad=True, device=DEVICE)
    log_gamma = torch.tensor(np.log(PARAM_INIT_GUESS[2]), requires_grad=True, device=DEVICE)
    log_rho = torch.tensor(np.log(PARAM_INIT_GUESS[3]), requires_grad=True, device=DEVICE)

    y0_scaled = torch.tensor(y0 / compartment_scale, device=DEVICE)
    net = build_network()
    forward = make_forward(net, y0_scaled)

    time_grid_warm = np.linspace(0, full_duration, 300)
    warm_trajectory = solve_seir(
        tuple(PARAM_INIT_GUESS[:3]), time_grid_warm, y0, population
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
    observed_scaled = torch.tensor(
        (observed / max(observed[-1], 1.0)).reshape(-1, 1), device=DEVICE
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
        rho = torch.exp(log_rho)

        r_s = ds + beta * s * i * (scale_I / N)
        r_e = de - beta * s * i * (scale_S * scale_I) / (N * scale_E) + sigma * e
        r_i = di - sigma * e * (scale_E / scale_I) + gamma * i
        r_r = dr - gamma * i * (scale_I / scale_R)

        physics_loss = (r_s ** 2).mean() + (r_e ** 2).mean() \
            + (r_i ** 2).mean() + (r_r ** 2).mean()

        y_data = forward(t_obs_tensor)
        simulated_observed_scaled = rho * (y_data[:, 2:3] * scale_I
                                            + y_data[:, 3:4] * scale_R) \
            / max(observed[-1], 1.0)
        data_loss = ((simulated_observed_scaled - observed_scaled) ** 2).mean()

        return physics_loss + 100 * data_loss

    optimizer = torch.optim.Adam(
        list(net.parameters()) + [log_beta, log_sigma, log_gamma, log_rho],
        lr=LEARNING_RATE,
    )
    for _ in range(N_ADAM_ITERATIONS):
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()

    lbfgs_optimizer = torch.optim.LBFGS(
        list(net.parameters()) + [log_beta, log_sigma, log_gamma, log_rho],
        lr=1.0, max_iter=N_LBFGS_ITERATIONS, line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs_optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        return loss

    lbfgs_optimizer.step(closure)

    fitted_params = np.array([
        float(torch.exp(log_beta.detach())),
        float(torch.exp(log_sigma.detach())),
        float(torch.exp(log_gamma.detach())),
        float(torch.exp(log_rho.detach())),
    ])

    with torch.no_grad():
        y_eval = forward(t_obs_tensor)
        simulated_observed = fitted_params[3] * (
            y_eval[:, 2].cpu().numpy() * scale_I
            + y_eval[:, 3].cpu().numpy() * scale_R
        )

    return fitted_params, simulated_observed


def fit_country(country_key):
    t_obs, observed, y0, population = load_country_data(country_key)
    fitted_params, simulated_observed = train_einn(t_obs, observed, y0, population)

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
             linewidth=2, label="EINN-fitted trajectory")
    ax.set_xlabel("Days since window start")
    ax.set_ylabel("Cumulative confirmed cases")
    ax.set_title(
        f"{display_name}: EINN fit to real first-wave data\n"
        f"beta={fit_result['beta']:.3f}, sigma={fit_result['sigma']:.3f}, "
        f"gamma={fit_result['gamma']:.3f}, rho={fit_result['rho']:.4f}, "
        f"relative RMSE={fit_result['rmse_relative']:.2%}"
    )
    ax.legend()
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"real_data_fit_einn_{fit_result['country']}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    summary_rows = []
    for country_key, config in COUNTRIES.items():
        print(f"\nFitting {config['display_name']} (EINN)...")
        fit_result = fit_country(country_key)
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
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "real_data_einn_summary.csv"),
                       index=False)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
