from dataclasses import dataclass

import numpy as np

from .config import ViscosityConfig
from .initial_conditions import apply_boundaries, initial_profile
from .schemes import step_advection_diffusion, step_upwind


@dataclass
class SimulationPair:
    x_grid: np.ndarray
    upwind_history: np.ndarray
    advection_diffusion_history: np.ndarray


def run_simulation(config: ViscosityConfig) -> SimulationPair:
    x_grid = config.x_grid()
    nx = x_grid.size
    nt = config.nt

    u_upwind = initial_profile(x_grid)
    u_advdiff = initial_profile(x_grid)
    apply_boundaries(u_upwind)
    apply_boundaries(u_advdiff)

    upwind_history = np.zeros((nt + 1, nx))
    advdiff_history = np.zeros((nt + 1, nx))
    upwind_history[0] = u_upwind.copy()
    advdiff_history[0] = u_advdiff.copy()

    for n in range(1, nt + 1):
        u_upwind = step_upwind(u_upwind, config.a, config.dt, config.dx)
        u_advdiff = step_advection_diffusion(
            u_advdiff,
            config.a,
            config.epsilon_num,
            config.dt,
            config.dx,
        )
        apply_boundaries(u_upwind)
        apply_boundaries(u_advdiff)
        upwind_history[n] = u_upwind.copy()
        advdiff_history[n] = u_advdiff.copy()

    return SimulationPair(
        x_grid=x_grid,
        upwind_history=upwind_history,
        advection_diffusion_history=advdiff_history,
    )
