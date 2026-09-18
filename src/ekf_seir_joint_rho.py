"""
ekf_seir_joint_rho.py

Extension of the Extended Kalman Filter in ekf_seir.py that estimates the
reporting fraction rho jointly with the epidemiological parameters,
rather than fixing it at an externally supplied value. This directly
addresses the rho-fixed asymmetry (relative to NLS and EINN) noted as a
limitation of the main real-data analysis (real_data_fit_ekf.py).

WHY THIS REQUIRES A SEPARATE OBSERVATION MODEL
---------------------------------------------------
In the main EKF module, rho is a fixed, externally supplied scalar, and
the observation model z = rho * (H . x) is LINEAR in the augmented state
x, so a fixed observation matrix H suffices and the standard linear
Kalman update applies directly. Once rho is instead included as an
eighth, filtered state component, the observation becomes
z = x_rho * (x_I + x_R), a PRODUCT of two state components, and is
therefore NONLINEAR in the augmented state. This module generalizes the
observation step to a nonlinear observation function h(x), with its
Jacobian obtained by the same finite-difference approach already used for
the state-transition Jacobian in ekf_seir.py, rather than a fixed matrix.
The state-transition step itself (the SEIR dynamics and the random-walk
model for beta, sigma, gamma) is unchanged from ekf_seir.py; rho is
appended as a ninth-like component evolving under the same random-walk
assumption as the other three parameters.

HOW TO USE
-----------
    from ekf_seir_joint_rho import run_ekf_joint_rho

DEPENDENCIES
-------------
numpy
"""

import numpy as np

from ekf_seir import seir_rhs, rk4_step


def state_transition_joint_rho(x, population, dt, n_substeps=4):
    """Advance the 8-dimensional augmented state
    [S, E, I, R, beta, sigma, gamma, rho] by one observation interval.
    rho, like beta, sigma, and gamma, is held constant over the
    deterministic part of the step (its process noise is added
    separately, outside this function).
    """
    S, E, I, R, beta, sigma, gamma, rho = x
    y = np.array([S, E, I, R])
    sub_dt = dt / n_substeps
    for _ in range(n_substeps):
        y = rk4_step(y, beta, sigma, gamma, population, sub_dt)
    return np.array([y[0], y[1], y[2], y[3], beta, sigma, gamma, rho])


def state_transition_jacobian_joint_rho(x, population, dt, eps_rel=1e-6):
    """Numerical Jacobian of `state_transition_joint_rho`, 8x8, via
    centered finite differences.
    """
    n = len(x)
    jac = np.zeros((n, n))
    for i in range(n):
        step = eps_rel * max(abs(x[i]), 1e-8)
        x_plus = x.copy()
        x_plus[i] += step
        x_minus = x.copy()
        x_minus[i] -= step
        f_plus = state_transition_joint_rho(x_plus, population, dt)
        f_minus = state_transition_joint_rho(x_minus, population, dt)
        jac[:, i] = (f_plus - f_minus) / (2 * step)
    return jac


def observation_function(x):
    """Nonlinear observation: rho * (I + R), with rho itself a state
    component (index 7) rather than an external constant.
    """
    I, R, rho = x[2], x[3], x[7]
    return np.array([rho * (I + R)])


def observation_jacobian(x, eps_rel=1e-6):
    """Numerical Jacobian of `observation_function` with respect to the
    8-dimensional augmented state, via centered finite differences.
    """
    n = len(x)
    base_dim = 1
    jac = np.zeros((base_dim, n))
    for i in range(n):
        step = eps_rel * max(abs(x[i]), 1e-8)
        x_plus = x.copy()
        x_plus[i] += step
        x_minus = x.copy()
        x_minus[i] -= step
        h_plus = observation_function(x_plus)
        h_minus = observation_function(x_minus)
        jac[:, i] = (h_plus - h_minus) / (2 * step)
    return jac


def run_ekf_joint_rho(observations, dt, y0, param_init, rho_init, population,
                       observation_noise_std=None, process_noise_std=None,
                       param_process_noise_std=None, rho_process_noise_std=1e-5,
                       initial_param_uncertainty=None, initial_rho_uncertainty=None):
    """Run the Extended Kalman Filter with an 8-dimensional augmented
    state [S, E, I, R, beta, sigma, gamma, rho], jointly estimating the
    reporting fraction rho alongside the epidemiological parameters, over
    a nonlinear observation z = rho * (I + R).

    Parameters
    ----------
    observations : array of observed cumulative confirmed cases.
    dt : observation interval, in days.
    y0 : initial SEIR compartment values [S0, E0, I0, R0].
    param_init : initial guess [beta, sigma, gamma].
    rho_init : initial guess for rho.
    population : total population size.
    rho_process_noise_std : process noise standard deviation for rho;
        kept small by default, since rho is expected to be much more
        stable over the observation window than the state compartments.

    Returns
    -------
    state_estimates : array, shape (n_steps, 8).
    param_estimate_final : array, [beta, sigma, gamma, rho] at the last step.
    """
    n_steps = len(observations)

    if observation_noise_std is None:
        observation_noise_std = np.maximum(0.10 * np.abs(observations), 1.0)
    elif np.isscalar(observation_noise_std):
        observation_noise_std = np.full(n_steps, observation_noise_std)

    if process_noise_std is None:
        process_noise_std = np.array([1.0, 1.0, 1.0, 1.0])
    if param_process_noise_std is None:
        param_process_noise_std = np.array([1e-4, 1e-4, 1e-4])
    if initial_param_uncertainty is None:
        initial_param_uncertainty = np.array([0.5, 0.5, 0.5]) * param_init
    if initial_rho_uncertainty is None:
        initial_rho_uncertainty = 0.5 * rho_init

    x = np.concatenate([y0, param_init, [rho_init]]).astype(float)
    P = np.diag(np.concatenate([
        (0.01 * np.array(y0) + 1.0) ** 2,
        initial_param_uncertainty ** 2,
        [initial_rho_uncertainty ** 2],
    ]))

    Q = np.diag(np.concatenate([
        process_noise_std ** 2,
        param_process_noise_std ** 2,
        [rho_process_noise_std ** 2],
    ]))

    state_estimates = np.zeros((n_steps, 8))

    for t in range(n_steps):
        if t == 0:
            x_pred = x
            P_pred = P
        else:
            F = state_transition_jacobian_joint_rho(x, population, dt)
            x_pred = state_transition_joint_rho(x, population, dt)
            x_pred[:4] = np.maximum(x_pred[:4], 0.0)
            x_pred[4:7] = np.maximum(x_pred[4:7], 1e-6)
            x_pred[7] = np.clip(x_pred[7], 1e-6, 1.0)
            P_pred = F @ P @ F.T + Q

        H_lin = observation_jacobian(x_pred)
        h_pred = observation_function(x_pred)

        R_obs = np.array([[observation_noise_std[t] ** 2]])
        y_residual = observations[t] - h_pred
        S_innovation = H_lin @ P_pred @ H_lin.T + R_obs
        K_gain = P_pred @ H_lin.T @ np.linalg.inv(S_innovation)

        x = x_pred + (K_gain @ y_residual).flatten()
        x[:4] = np.maximum(x[:4], 0.0)
        x[4:7] = np.maximum(x[4:7], 1e-6)
        x[7] = np.clip(x[7], 1e-6, 1.0)
        P = (np.eye(8) - K_gain @ H_lin) @ P_pred

        state_estimates[t] = x

    param_estimate_final = np.concatenate([state_estimates[-1, 4:7],
                                            [state_estimates[-1, 7]]])
    return state_estimates, param_estimate_final
