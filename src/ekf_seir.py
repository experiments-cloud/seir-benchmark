"""
ekf_seir.py

Extended Kalman Filter (EKF) for joint state-and-parameter estimation in
the SEIR epidemic model. Shared module used by the EKF-based benchmark
scripts, analogous in role to seir_model.py for the ODE forward solver.

APPROACH
---------
The filter's state vector is augmented with the three epidemiological
parameters, which are modeled as constant in continuous time but are
assigned a small process noise so the filter can adapt them as data
arrives (a standard "parameter as augmented state with random-walk
process noise" formulation, common in engineering applications of the
EKF to parameter estimation):

    x = [S, E, I, R, beta, sigma, gamma]

The (nonlinear) state transition is the SEIR ODE for the first four
components and a random walk (identity map plus process noise) for the
three parameter components. Discretization uses a fixed-step 4th-order
Runge-Kutta integration of the continuous dynamics between observation
times, since the observation interval (1 day) is coarse relative to the
epidemic's timescale and a first-order (Euler) discretization would
introduce non-negligible integration error into the filter.

The observation model assumes only the infectious compartment I is
observed (optionally scaled by an underreporting fraction rho), which is
linear in the state, so the observation Jacobian is a fixed selection
(and scaling) matrix.

HOW TO USE
-----------
    from ekf_seir import run_ekf

DEPENDENCIES
-------------
numpy
"""

import numpy as np


def seir_rhs(y, beta, sigma, gamma, population):
    S, E, I, R = y
    dS = -beta * S * I / population
    dE = beta * S * I / population - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return np.array([dS, dE, dI, dR])


def rk4_step(y, beta, sigma, gamma, population, dt):
    """Single fixed-step 4th-order Runge-Kutta integration step for the
    SEIR compartments only (the parameter components of the augmented
    state are handled separately, as a random walk, by the caller).
    """
    k1 = seir_rhs(y, beta, sigma, gamma, population)
    k2 = seir_rhs(y + 0.5 * dt * k1, beta, sigma, gamma, population)
    k3 = seir_rhs(y + 0.5 * dt * k2, beta, sigma, gamma, population)
    k4 = seir_rhs(y + dt * k3, beta, sigma, gamma, population)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def state_transition(x, population, dt, n_substeps=4):
    """Advance the augmented state [S, E, I, R, beta, sigma, gamma] by one
    observation interval `dt`, using `n_substeps` RK4 steps internally for
    numerical accuracy. Parameters are held constant over the step (their
    process noise is added separately, outside this function).
    """
    S, E, I, R, beta, sigma, gamma = x
    y = np.array([S, E, I, R])
    sub_dt = dt / n_substeps
    for _ in range(n_substeps):
        y = rk4_step(y, beta, sigma, gamma, population, sub_dt)
    return np.array([y[0], y[1], y[2], y[3], beta, sigma, gamma])


def state_transition_jacobian(x, population, dt, eps_rel=1e-6):
    """Numerical Jacobian of `state_transition` with respect to the
    augmented state, via centered finite differences. Used for covariance
    propagation in the EKF prediction step.
    """
    n = len(x)
    jac = np.zeros((n, n))
    base = state_transition(x, population, dt)
    for i in range(n):
        step = eps_rel * max(abs(x[i]), 1e-8)
        x_plus = x.copy()
        x_plus[i] += step
        x_minus = x.copy()
        x_minus[i] -= step
        f_plus = state_transition(x_plus, population, dt)
        f_minus = state_transition(x_minus, population, dt)
        jac[:, i] = (f_plus - f_minus) / (2 * step)
    return jac


def run_ekf(observations, dt, y0, param_init, population, rho=1.0,
            observation_noise_std=None, process_noise_std=None,
            param_process_noise_std=None, initial_param_uncertainty=None,
            observation_weights=None):
    """Run the Extended Kalman Filter over a sequence of scalar
    observations, recovering both the compartment trajectory and the
    epidemiological parameters.

    Parameters
    ----------
    observations : array of observed (possibly noisy, possibly
        underreported) case counts, one per time step.
    dt : observation interval, in days.
    y0 : initial SEIR compartment values [S0, E0, I0, R0] (assumed known
        exactly, as is standard practice: population accounting is usually
        reliable at t=0 even when case counts are not).
    param_init : initial guess [beta, sigma, gamma] for the filter.
    population : total population size.
    rho : underreporting/reporting fraction, applied as a scalar
        multiplier on the observed quantity (observations = rho * (H . x)
        + noise, where H is defined by `observation_weights`).
    observation_noise_std : array or scalar, standard deviation of the
        observation noise at each time step. If None, a default constant
        value is used.
    process_noise_std : array of length 4, process noise standard
        deviation for the SEIR compartments (reflects model/discretization
        uncertainty). If None, a small default is used.
    param_process_noise_std : array of length 3, process noise standard
        deviation for beta, sigma, gamma (controls how much the filter is
        allowed to adapt the parameters over time; smaller values make the
        parameter estimate more stable but slower to adapt).
    initial_param_uncertainty : array of length 3, initial standard
        deviation assigned to the parameter estimates.
    observation_weights : array of length 4, optional. Defines which
        linear combination of the SEIR compartments [S, E, I, R] is
        observed, before applying `rho`. Defaults to [0, 0, 1, 0] (observe
        I only, matching the synthetic-data benchmark scripts). For real
        cumulative case-count data, pass [0, 0, 1, 1] to observe I + R
        (see real_data_fit_ekf.py for that use case).

    Returns
    -------
    state_estimates : array, shape (n_steps, 7), the filtered state
        (S, E, I, R, beta, sigma, gamma) at each time step.
    param_estimate_final : array, [beta, sigma, gamma] at the last step.
    """
    n_steps = len(observations)

    if observation_noise_std is None:
        observation_noise_std = np.maximum(0.10 * np.abs(observations), 1.0)
    elif np.isscalar(observation_noise_std):
        observation_noise_std = np.full(n_steps, observation_noise_std)

    if process_noise_std is None:
        process_noise_std = np.array([1.0, 1.0, 1.0, 1.0])
    if param_process_noise_std is None:
        # Calibrated by a small tuning sweep during development (values
        # 1e-4, 1e-3, 1e-2, 5e-2 tested against clean, full-length data):
        # 1e-4 gave the lowest parameter-recovery error. Larger values let
        # the filter adapt faster but also let it drift away from the
        # correct parameters under noise; smaller values were not tested
        # further since 1e-4 was already close to the lower end of the
        # sweep and performed best.
        param_process_noise_std = np.array([1e-4, 1e-4, 1e-4])
    if initial_param_uncertainty is None:
        initial_param_uncertainty = np.array([0.5, 0.5, 0.5]) * param_init
    if observation_weights is None:
        observation_weights = np.array([0.0, 0.0, 1.0, 0.0])

    x = np.concatenate([y0, param_init]).astype(float)
    P = np.diag(np.concatenate([
        (0.01 * np.array(y0) + 1.0) ** 2,   # small initial state uncertainty
        initial_param_uncertainty ** 2,
    ]))

    Q = np.diag(np.concatenate([
        process_noise_std ** 2,
        param_process_noise_std ** 2,
    ]))

    # Observation model: z = rho * (observation_weights . x) + noise.
    H = np.zeros((1, 7))
    H[0, :4] = rho * observation_weights

    state_estimates = np.zeros((n_steps, 7))

    for t in range(n_steps):
        if t == 0:
            x_pred = x
            P_pred = P
        else:
            F = state_transition_jacobian(x, population, dt)
            x_pred = state_transition(x, population, dt)
            # Keep predicted state physically valid (non-negative
            # compartments, positive parameters) to avoid the filter
            # diverging into an unphysical region.
            x_pred[:4] = np.maximum(x_pred[:4], 0.0)
            x_pred[4:] = np.maximum(x_pred[4:], 1e-6)
            P_pred = F @ P @ F.T + Q

        R_obs = np.array([[observation_noise_std[t] ** 2]])
        y_residual = observations[t] - (H @ x_pred)
        S_innovation = H @ P_pred @ H.T + R_obs
        K_gain = P_pred @ H.T @ np.linalg.inv(S_innovation)

        x = x_pred + (K_gain @ y_residual).flatten()
        x[:4] = np.maximum(x[:4], 0.0)
        x[4:] = np.maximum(x[4:], 1e-6)
        P = (np.eye(7) - K_gain @ H) @ P_pred

        state_estimates[t] = x

    param_estimate_final = state_estimates[-1, 4:]
    return state_estimates, param_estimate_final
