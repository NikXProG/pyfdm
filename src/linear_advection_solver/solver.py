from dataclasses import dataclass

import numpy as np

from .config import AdvectionConfig
from .initial_conditions import apply_left_boundary, initial_profile
from .schemes import StepScheme


@dataclass
class SimulationResult:
    scheme_name: str
    x_grid: np.ndarray
    u_final: np.ndarray
    u_history: np.ndarray


def solve(config: AdvectionConfig, scheme_name: str, step_fn: StepScheme) -> SimulationResult:
    x_grid = config.x_grid()
    nu = config.a * config.dt / config.dx

    u = initial_profile(x_grid)
    apply_left_boundary(u, 0.0)
    u_history = np.zeros((config.nt + 1, x_grid.size))
    u_history[0] = u.copy()

    for n in range(1, config.nt + 1):
        u_new = step_fn(u, nu)
        # На первых узлах для высоких схем аккуратно добираем устойчивым upwind шагом.
        u_new[1] = u[1] - nu * (u[1] - u[0])
        apply_left_boundary(u_new, 0.0)
        u = u_new
        u_history[n] = u.copy()

    return SimulationResult(scheme_name=scheme_name, x_grid=x_grid, u_final=u, u_history=u_history)
