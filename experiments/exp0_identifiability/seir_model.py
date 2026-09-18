"""
seir_model.py

Shared SEIR (Susceptible-Exposed-Infectious-Recovered) model utilities used
across the experiments in this benchmark. Kept separate from any single
experiment script so it can be imported and reused without duplication.

HOW TO USE
-----------
    from seir_model import solve_seir, determine_outbreak_duration

DEPENDENCIES
-------------
numpy, scipy
"""

import numpy as np
from scipy.integrate import solve_ivp


def seir_rhs(t, y, beta, sigma, gamma, population):
    """Right-hand side of the SEIR ODE system.

    Parameters
    ----------
    beta : transmission rate
    sigma : incubation rate (1 / mean incubation period)
    gamma : recovery rate (1 / mean infectious period)
    population : total population size (S + E + I + R)
    """
    S, E, I, R = y
    dS = -beta * S * I / population
    dE = beta * S * I / population - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


def solve_seir(params, t_eval, y0, population, rtol=1e-9, atol=1e-11):
    """Integrate the SEIR system for a given parameter set over the
    provided time grid `t_eval`.

    Parameters
    ----------
    params : sequence of (beta, sigma, gamma)
    t_eval : array of time points at which to report the solution
    y0 : initial condition [S0, E0, I0, R0]
    population : total population size
    rtol, atol : integrator tolerances. The defaults are tight, appropriate
        for generating reference/ground-truth trajectories. Looser
        tolerances (e.g. 1e-6 / 1e-9) are recommended when this function is
        called repeatedly inside an optimization loop, to reduce runtime
        with a negligible effect on fitted-parameter accuracy.

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
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return solution.y


def determine_outbreak_duration(
    params,
    y0,
    population,
    search_horizon_days=730,
    decline_fraction=0.01,
    buffer_days=15,
):
    """Determine how many days are needed to observe a full epidemic cycle
    (growth, peak, and decline of the infectious compartment) for a given
    parameter set, instead of assuming a fixed, arbitrary horizon.

    The outbreak is considered "resolved" once the infectious compartment
    drops below `decline_fraction` of its peak value, after having passed
    the peak. A fixed buffer (in days) is added so the curve visibly
    flattens out.

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
            "search_horizon_days. Falling back to the full search horizon."
        )
        return float(search_horizon_days)

    end_index = peak_index + below_threshold[0]
    end_day = search_grid[end_index] + buffer_days
    return float(min(end_day, search_horizon_days))
