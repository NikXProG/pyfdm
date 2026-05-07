from dataclasses import dataclass

import numpy as np

from .config import RecalculationHeatConfig
from .initial_conditions import apply_boundaries, initial_profile
from .schemes import full_step_recalculation, half_step_implicit


@dataclass
class RecalculationResult:
    x_grid: np.ndarray
    t_grid: np.ndarray
    u_history: np.ndarray


def solve(config: RecalculationHeatConfig) -> RecalculationResult:
    x = config.x_grid()
    t = config.t_grid()
    u = initial_profile(x)
    apply_boundaries(u)

    history = np.zeros((t.size, x.size))
    history[0] = u.copy()

    for n in range(1, t.size):
        u_half = half_step_implicit(u, config.dt, config.dx)
        u_new = full_step_recalculation(u, u_half, config.dt, config.dx)
        apply_boundaries(u_new)
        u = u_new
        history[n] = u.copy()

    return RecalculationResult(x_grid=x, t_grid=t, u_history=history)
