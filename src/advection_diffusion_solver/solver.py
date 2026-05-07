from dataclasses import dataclass

import numpy as np

from .config import GridConfig, Scenario
from .initial_conditions import apply_dirichlet_boundaries, box_pulse_initial
from .schemes import UxScheme, uxx_central


@dataclass
class SimulationResult:
    x_grid: np.ndarray
    t_grid: np.ndarray
    u_history: np.ndarray
    scenario: Scenario
    ux_scheme_name: str


def solve_explicit_euler(
    grid: GridConfig,
    scenario: Scenario,
    ux_scheme_name: str,
    ux_scheme: UxScheme,
) -> SimulationResult:
    x_grid = grid.x_grid()
    t_grid = grid.t_grid()
    nx = x_grid.size
    nt = t_grid.size

    u = box_pulse_initial(x_grid)
    apply_dirichlet_boundaries(u)

    u_history = np.zeros((nt, nx))
    u_history[0] = u.copy()

    for n in range(1, nt):
        ux = ux_scheme(u, grid.dx)
        uxx = uxx_central(u, grid.dx)
        u_new = u + grid.dt * (-scenario.beta * ux + scenario.alpha * uxx)
        apply_dirichlet_boundaries(u_new)
        u = u_new
        u_history[n] = u.copy()

    return SimulationResult(
        x_grid=x_grid,
        t_grid=t_grid,
        u_history=u_history,
        scenario=scenario,
        ux_scheme_name=ux_scheme_name,
    )
