import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from seir_model import determine_outbreak_duration, solve_seir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

POPULATION_SIZE = 1_000_000
BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE = 0.9, 1/8.0, 0.1
PARAMS_TRUE = np.array([BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE])
E0,I0,R0 = 1,1,0
S0 = POPULATION_SIZE-E0-I0-R0
Y0 = np.array([S0,E0,I0,R0], dtype=float)

NOISE_LEVELS = [0.0, 0.30, 1.00]
N_SEEDS = 3
N_COLLOCATION_POINTS = 1000
N_WARMUP_ITERATIONS = 1500
N_ADAM_ITERATIONS = 4000
N_LBFGS_ITERATIONS = 500
LEARNING_RATE = 1e-3
NETWORK_HIDDEN_LAYERS = [32,32,32]
PARAM_BOUNDS_LOWER = np.array([0.01, 1/14, 1/21])
PARAM_BOUNDS_UPPER = np.array([2.00, 1/2, 1/3])
RANDOM_SEED_BASE = 12345

OUTPUT_DIR = "results_noise_regime2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "einn_noise_regime2_raw_results.csv")

def generate_full_trajectory():
    full_duration = determine_outbreak_duration(PARAMS_TRUE, Y0, POPULATION_SIZE)
    n_points = int(np.ceil(full_duration/1.0))+1
    time_grid = np.linspace(0, full_duration, n_points)
    trajectory = solve_seir(PARAMS_TRUE, time_grid, Y0, POPULATION_SIZE)
    return time_grid, trajectory, full_duration

def build_network():
    layers=[]
    sizes=[1]+NETWORK_HIDDEN_LAYERS+[4]
    for i in range(len(sizes)-2):
        layers.append(torch.nn.Linear(sizes[i], sizes[i+1])); layers.append(torch.nn.Tanh())
    layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
    return torch.nn.Sequential(*layers).to(DEVICE)

def make_forward(net, y0_scaled):
    def forward(t):
        raw = net(t); factor = 1-torch.exp(-t)
        return y0_scaled + factor*raw
    return forward

def train_einn(t_obs, i_obs_noisy, full_duration, y0, population, compartment_scale, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(RANDOM_SEED_BASE+seed)
    log_p = rng.uniform(np.log(PARAM_BOUNDS_LOWER), np.log(PARAM_BOUNDS_UPPER))
    beta_i, sigma_i, gamma_i = np.exp(log_p)
    log_beta = torch.tensor(np.log(beta_i), requires_grad=True, device=DEVICE)
    log_sigma = torch.tensor(np.log(sigma_i), requires_grad=True, device=DEVICE)
    log_gamma = torch.tensor(np.log(gamma_i), requires_grad=True, device=DEVICE)
    scale_S,scale_E,scale_I,scale_R = compartment_scale
    y0_scaled = torch.tensor(y0/compartment_scale, device=DEVICE)
    net = build_network(); forward = make_forward(net, y0_scaled)

    twarm = np.linspace(0, full_duration, 300)
    warm_traj = solve_seir((beta_i,sigma_i,gamma_i), twarm, y0, population)
    warm_target = torch.tensor((warm_traj/compartment_scale.reshape(-1,1)).T, device=DEVICE)
    t_warm = torch.tensor(twarm.reshape(-1,1), device=DEVICE)
    wopt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    for _ in range(N_WARMUP_ITERATIONS):
        wopt.zero_grad(); pred=forward(t_warm); l=((pred-warm_target)**2).mean(); l.backward(); wopt.step()

    coll_t = torch.linspace(0, full_duration, N_COLLOCATION_POINTS, device=DEVICE).reshape(-1,1).requires_grad_(True)
    t_obs_t = torch.tensor(t_obs.reshape(-1,1), device=DEVICE)
    i_obs_scaled = torch.tensor((i_obs_noisy/scale_I).reshape(-1,1), device=DEVICE)
    N=population

    def compute_loss():
        y_col = forward(coll_t)
        s,e,i,r = y_col[:,0:1],y_col[:,1:2],y_col[:,2:3],y_col[:,3:4]
        go = torch.ones_like(s)
        ds=torch.autograd.grad(s,coll_t,grad_outputs=go,create_graph=True)[0]
        de=torch.autograd.grad(e,coll_t,grad_outputs=go,create_graph=True)[0]
        di=torch.autograd.grad(i,coll_t,grad_outputs=go,create_graph=True)[0]
        dr=torch.autograd.grad(r,coll_t,grad_outputs=go,create_graph=True)[0]
        beta=torch.exp(log_beta); sigma=torch.exp(log_sigma); gamma=torch.exp(log_gamma)
        r_s = ds + beta*s*i*(scale_I/N)
        r_e = de - beta*s*i*(scale_S*scale_I)/(N*scale_E) + sigma*e
        r_i = di - sigma*e*(scale_E/scale_I) + gamma*i
        r_r = dr - gamma*i*(scale_I/scale_R)
        physics = (r_s**2).mean()+(r_e**2).mean()+(r_i**2).mean()+(r_r**2).mean()
        y_data = forward(t_obs_t)
        data_loss = ((y_data[:,2:3]-i_obs_scaled)**2).mean()
        return physics + 100*data_loss

    opt = torch.optim.Adam(list(net.parameters())+[log_beta,log_sigma,log_gamma], lr=LEARNING_RATE)
    for _ in range(N_ADAM_ITERATIONS):
        opt.zero_grad(); loss=compute_loss(); loss.backward(); opt.step()

    lbfgs = torch.optim.LBFGS(list(net.parameters())+[log_beta,log_sigma,log_gamma], lr=1.0, max_iter=N_LBFGS_ITERATIONS, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad(); loss=compute_loss(); loss.backward(); return loss
    lbfgs.step(closure)

    fitted = np.array([float(torch.exp(log_beta.detach())), float(torch.exp(log_sigma.detach())), float(torch.exp(log_gamma.detach()))])
    return fitted

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return pd.read_csv(CHECKPOINT_PATH)
    return pd.DataFrame(columns=["noise_level","seed","beta_fitted","sigma_fitted","gamma_fitted","mean_rel_error"])

def append_checkpoint(row):
    df = pd.DataFrame([row])
    write_header = not os.path.exists(CHECKPOINT_PATH)
    df.to_csv(CHECKPOINT_PATH, mode="a", header=write_header, index=False)

def main(max_fits_this_call=3):
    time_grid, full_traj, full_duration = generate_full_trajectory()
    infectious = full_traj[2]
    compartment_scale = np.maximum(np.max(np.abs(full_traj), axis=1), 1.0)

    checkpoint = load_checkpoint()
    done_pairs = set(zip(checkpoint["noise_level"], checkpoint["seed"])) if len(checkpoint) else set()

    all_jobs = [(nl, seed) for nl in NOISE_LEVELS for seed in range(N_SEEDS)]
    n_done_this_call = 0
    for nl, seed in all_jobs:
        if (nl, seed) in done_pairs:
            continue
        if n_done_this_call >= max_fits_this_call:
            print(f"Reached this call's budget; {len(done_pairs)+n_done_this_call}/{len(all_jobs)} total done.")
            return
        rng = np.random.default_rng(RANDOM_SEED_BASE+seed+int(nl*1000))
        noise_std = np.maximum(nl*np.abs(infectious), 1.0)
        i_obs_noisy = np.clip(infectious + rng.normal(0.0, noise_std), 0.0, None)
        fitted = train_einn(time_grid, i_obs_noisy, full_duration, Y0, POPULATION_SIZE, compartment_scale, seed=RANDOM_SEED_BASE+seed)
        rel_err = np.abs(fitted-PARAMS_TRUE)/PARAMS_TRUE
        row = {"noise_level":nl, "seed":seed, "beta_fitted":fitted[0], "sigma_fitted":fitted[1],
               "gamma_fitted":fitted[2], "mean_rel_error": float(np.mean(rel_err))}
        append_checkpoint(row)
        n_done_this_call += 1
        print(f"DONE noise={nl:.0%} seed={seed}: mean_rel_error={row['mean_rel_error']:.4f} "
              f"({len(done_pairs)+n_done_this_call}/{len(all_jobs)} total)")

    print(f"ALL {len(all_jobs)} FITS COMPLETE.")

if __name__ == "__main__":
    main(max_fits_this_call=int(os.environ.get("MAX_FITS", 3)))
