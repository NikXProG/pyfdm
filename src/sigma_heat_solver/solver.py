from dataclasses import dataclass

import numpy as np

from .config import SigmaHeatConfig
from .initial_conditions import apply_boundaries, initial_profile
from .schemes import sigma_step


@dataclass
class SigmaRun:
    sigma: float
    x_grid: np.ndarray
    t_grid: np.ndarray
    u_history: np.ndarray


def solve_for_sigma(config: SigmaHeatConfig, sigma: float) -> SigmaRun:
    x = config.x_grid()
    t = config.t_grid()
    u = initial_profile(x)
    apply_boundaries(u)

    history = np.zeros((t.size, x.size))
    history[0] = u.copy()

    for n in range(1, t.size):
        u_new = sigma_step(u, sigma, config.r, config.dx)
        apply_boundaries(u_new)
        u = u_new
        history[n] = u.copy()

    return SigmaRun(sigma=sigma, x_grid=x, t_grid=t, u_history=history)


def solve_all(config: SigmaHeatConfig) -> list[SigmaRun]:
    return [solve_for_sigma(config, sigma) for sigma in config.sigma_values]
