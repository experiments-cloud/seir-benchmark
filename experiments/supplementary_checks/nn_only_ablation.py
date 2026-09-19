"""
nn_only_ablation.py

Ablation for Section 4.5 (real-data validation): trains a plain feed-
forward neural network directly against each country's cumulative case
curve, with no SEIR structure, no physics-residual loss term, and no
epidemiological parameters at all, just curve-fitting. The question this
isolates: does EINN's advantage on real data (Table 8) come from the
physics constraint specifically, or would a similarly flexible network
fit the misspecified curve about as well without it?

Same network architecture as EINN (three hidden layers of 32 units,
tanh activations; Methods 3.3) and a comparable training budget (4000
Adam iterations, up to 500 L-BFGS iterations), so any difference in fit
quality is attributable to the presence or absence of the physics
constraint rather than to network capacity or training length. Time is
rescaled to [0, 1] and the case count is rescaled by its own final value,
for numerical conditioning only, exactly as EINN rescales each
compartment by its own peak magnitude.

Reports the same relative RMSE metric used in Table 8 (computed on days
with at least 50 cumulative cases), for direct comparison against EINN's
42.6% (Italy) and 4.7% (South Korea).

Requires: numpy, pandas, torch. Runs in well under a minute per country
on CPU, no GPU needed, since there is no ODE integration or automatic
differentiation through physics residuals, only a direct regression fit.
"""

import numpy as np
import pandas as pd
import torch

DEVICE = torch.device("cpu")
torch.set_default_dtype(torch.float64)

NETWORK_HIDDEN_LAYERS = [32, 32, 32]
N_ADAM_ITERATIONS = 4000
N_LBFGS_ITERATIONS = 500
LEARNING_RATE = 1e-3
RANDOM_SEED = 12345

COUNTRIES = {
    "italy": "real_data_italy.csv",
    "korea": "real_data_korea.csv",
}


def build_network():
    layers = []
    sizes = [1] + NETWORK_HIDDEN_LAYERS + [1]
    for i in range(len(sizes) - 2):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(torch.nn.Tanh())
    layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
    return torch.nn.Sequential(*layers).to(DEVICE)


def fit_nn_only(t_obs, observed):
    torch.manual_seed(RANDOM_SEED)

    t_scale = t_obs[-1]
    c_scale = observed[-1]
    t_scaled = torch.tensor((t_obs / t_scale).reshape(-1, 1), device=DEVICE)
    c_scaled = torch.tensor((observed / c_scale).reshape(-1, 1), device=DEVICE)

    net = build_network()

    def compute_loss():
        pred = net(t_scaled)
        return ((pred - c_scaled) ** 2).mean()

    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    for _ in range(N_ADAM_ITERATIONS):
        opt.zero_grad()
        loss = compute_loss()
        loss.backward()
        opt.step()

    lbfgs = torch.optim.LBFGS(
        net.parameters(), lr=1.0, max_iter=N_LBFGS_ITERATIONS,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        loss = compute_loss()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        fitted_scaled = net(t_scaled).cpu().numpy().flatten()
    return fitted_scaled * c_scale


def main():
    results = []
    for country_key, filename in COUNTRIES.items():
        df = pd.read_csv(filename)
        t_obs = df["day"].values.astype(float)
        observed = df["cumulative_confirmed"].values.astype(float)

        fitted = fit_nn_only(t_obs, observed)

        valid = observed >= 50
        rel_res = (fitted - observed)[valid] / np.maximum(observed[valid], 10.0)
        rmse = float(np.sqrt(np.mean(rel_res ** 2)))

        results.append({"country": country_key, "relative_rmse": rmse})
        print(f"{country_key}: relative RMSE = {rmse:.2%}")

        out_df = pd.DataFrame({"day": t_obs, "observed": observed, "nn_only_fitted": fitted})
        out_df.to_csv(f"nn_only_fit_{country_key}.csv", index=False)

    summary = pd.DataFrame(results)
    summary.to_csv("nn_only_ablation_summary.csv", index=False)
    print()
    print("Saved per-day fits to nn_only_fit_<country>.csv and summary to "
          "nn_only_ablation_summary.csv")
    print()
    print("For comparison, EINN's relative RMSE (Table 8): Italy 42.6%, South Korea 4.7%.")


if __name__ == "__main__":
    main()
