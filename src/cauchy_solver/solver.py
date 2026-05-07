from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .physics import diffusion_like_coeff, initial_displacement, initial_velocity, reaction_coeff
from .schemes import build_spatial_operator_coeffs
from solver_utils import solve_tridiagonal_system


@dataclass
class SimulationResult:
    x_grid: np.ndarray
    t_grid: np.ndarray
    u_history: np.ndarray


def solve(config: SimulationConfig) -> SimulationResult:
    x_grid = config.x_grid()
    t_grid = config.t_grid()
    nx = x_grid.size
    nt = t_grid.size - 1
    dt2_inv = 1.0 / (config.dt * config.dt)

    d = diffusion_like_coeff(x_grid)
    c = reaction_coeff(x_grid)
    lower_l, diag_l, upper_l = build_spatial_operator_coeffs(x_grid, d, c, config.dx)

    lower_a = -lower_l
    diag_a = dt2_inv - diag_l
    upper_a = -upper_l
    u_history = np.zeros((nt + 1, nx))
    u0 = initial_displacement(x_grid)
    v0 = initial_velocity(x_grid)
    u0[0] = 0.0
    u0[-1] = 0.0
    u_history[0] = u0.copy()

    u_prev = u0 - config.dt * v0
    u_prev[0] = 0.0
    u_prev[-1] = 0.0
    u_curr = u0.copy()

    rhs_first = dt2_inv * (u_curr[1:-1] + config.dt * v0[1:-1])
    inner_first = solve_tridiagonal_system(lower_a, diag_a, upper_a, rhs_first)
    u_next = np.zeros_like(u_curr)
    u_next[1:-1] = inner_first
    u_history[1] = u_next.copy()

    u_prev = u_curr
    u_curr = u_next

    for n in range(1, nt):
        rhs = dt2_inv * (2.0 * u_curr[1:-1] - u_prev[1:-1])
        inner = solve_tridiagonal_system(lower_a, diag_a, upper_a, rhs)
        u_next = np.zeros_like(u_curr)
        u_next[1:-1] = inner
        u_history[n + 1] = u_next.copy()
        u_prev = u_curr
        u_curr = u_next

    return SimulationResult(x_grid=x_grid, t_grid=t_grid, u_history=u_history)
