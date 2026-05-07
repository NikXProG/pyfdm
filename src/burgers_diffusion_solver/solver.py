from dataclasses import dataclass

import numpy as np

from .config import BurgersConfig
from .initial_conditions import apply_dirichlet_boundaries, initial_profile
from .schemes import explicit_step
from .time_step import safe_dt


@dataclass
class BurgersResult:
    x_grid: np.ndarray
    t_grid: np.ndarray
    u_history: np.ndarray


def solve_single_epsilon(config: BurgersConfig, epsilon: float) -> BurgersResult:
    x = config.x_grid()
    u = initial_profile(x)
    apply_dirichlet_boundaries(u)

    t_values = [0.0]
    history = [u.copy()]
    t = 0.0

    while t < config.t_final - 1e-12:
        dt = safe_dt(u, epsilon, config.dx, config.cfl_conv, config.cfl_diff)
        dt = min(dt, config.t_final - t)
        u_new = explicit_step(u, epsilon, dt, config.dx)
        apply_dirichlet_boundaries(u_new)
        u = u_new
        t += dt
        t_values.append(t)
        history.append(u.copy())

    return BurgersResult(x_grid=x, t_grid=np.array(t_values), u_history=np.array(history))


def sweep_epsilons(config: BurgersConfig, eps_values: np.ndarray) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    for eps in eps_values:
        result = solve_single_epsilon(config, float(eps))
        out[float(eps)] = result.u_history[-1]
    return out
