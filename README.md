# A Systematic Benchmark of Classical, Filtering, and Physics-Informed Estimators for SEIR Parameter Inference Under Data Sparsity, Noise, and Structural Non-Identifiability

A Systematic Benchmark for SEIR Parameter Inference Under Sparsity, Noise, and Structural Non-Identifiability

Code, data, and manuscript source for the manuscript submitted to *Mathematical Biosciences*. This repository provides everything needed to reproduce every number, table, and figure reported in the paper.

## Overview

We compare four parameter estimation methods for the SEIR epidemic model — nonlinear least squares (NLS), an Extended Kalman Filter (EKF), an epidemiology-informed neural network (EINN), and Bayesian Markov Chain Monte Carlo (MCMC) — across five experiments:

| Experiment | Folder | Methods |
|---|---|---|
| Identifiability diagnosis (Fisher Information Matrix) | `experiments/exp0_identifiability/` | diagnostic only |
| Parameter recovery under data sparsity | `experiments/exp1_sparsity/` | NLS, EKF, EINN |
| Parameter recovery under observation noise | `experiments/exp2_noise/` | NLS, EKF, EINN |
| Validation of the identifiability diagnosis | `experiments/exp3_identifiability_validation/` | NLS, EKF, EINN, MCMC |
| Real-world validation (COVID-19, Italy & South Korea) | `experiments/exp4_real_data/` | NLS, EKF, EINN, MCMC |

Supplementary robustness checks (a second parameter regime, an EINN hyperparameter ablation, a joint-ρ extension of the EKF, and an algebraic structural identifiability analysis in Julia) are in `experiments/supplementary_checks/`.

## Data

Real COVID-19 case counts for Italy and South Korea
(`experiments/exp4_real_data/real_data_italy.csv`,
`real_data_korea.csv`) are included directly in this repository, rather
than only documenting how to re-download them. Both files are small
(a few KB) and were originally obtained from the Johns Hopkins University
CSSE repository, which was archived (set to read-only) in March 2023; a
static local copy protects the exact reproducibility of the results
reported in the manuscript against that source becoming harder to access
in its original form. `download_real_data.py` is still provided for
reference and shows exactly how these files were produced from the
archived source.

No synthetic datasets are included, since every synthetic experiment
regenerates its own data deterministically from the fixed random seeds
documented in each script; running any experiment script reproduces its
inputs and outputs from scratch.

## Repository structure

```
seir-benchmark/
├── src/                              Canonical copies of the shared modules
│   ├── seir_model.py                 SEIR ODE solver, outbreak-duration finder
│   ├── ekf_seir.py                   Extended Kalman Filter implementation
│   └── ekf_seir_joint_rho.py         EKF variant with jointly estimated rho
├── experiments/
│   ├── exp0_identifiability/         Fisher Information Matrix diagnostic
│   ├── exp1_sparsity/                Experiment 1: data sparsity
│   ├── exp2_noise/                   Experiment 2: observation noise
│   ├── exp3_identifiability_validation/  Experiment 3: identifiability validation
│   ├── exp4_real_data/               Experiment 4: real COVID-19 data (Italy, South Korea)
│   └── supplementary_checks/         Second regime (sparsity + noise), EINN ablation, structural identifiability (generic and known-IC, Julia)
├── requirements.txt
└── LICENSE
```

Each experiment folder is self-contained: it includes its own copy of `seir_model.py` (and `ekf_seir.py` where needed) alongside the experiment scripts, so it can be run directly without setting `PYTHONPATH`.

## Reproducing the results

```bash
pip install -r requirements.txt
```

Each script writes its results (CSV summaries and PNG figures) to a `results/` subfolder created alongside it. All random seeds are fixed and documented in each script, so results are exactly reproducible. Run order within `exp1`–`exp3` does not matter between methods (NLS, EKF, EINN, MCMC are independent), but `exp4_real_data` requires `download_real_data.py` and `real_data_fit_nls.py` to be run first, since the EKF and MCMC scripts there reuse the reporting fraction already estimated by NLS.

The Bayesian MCMC scripts (`*_mcmc.py`) are markedly more expensive than the other three methods (on the order of tens of minutes to hours per fit); see the manuscript's Methods and Limitations sections for the reasons this experiment uses a reduced number of random initializations relative to NLS, EKF, and EINN.

The structural identifiability check (`experiments/supplementary_checks/structural_identifiability_seir.jl`) requires a separate Julia installation with the `StructuralIdentifiability.jl` package; see the comments at the top of that file.

## License

This repository is released under the MIT License (see `LICENSE`). See the Data section above for the provenance of the included real-world case counts.
