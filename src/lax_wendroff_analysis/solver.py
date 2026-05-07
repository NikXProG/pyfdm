from dataclasses import dataclass

import numpy as np

from .config import LWConfig
from .initial_conditions import initial_profile
from .schemes import step_lax_wendroff


@dataclass
class LWSimulation:
    mu: float
    nu: float
    x_grid: np.ndarray
    u0: np.ndarray
    u_final: np.ndarray


def exact_shift_periodic(u0: np.ndarray, shift_x: float, dx: float) -> np.ndarray:
    shift_cells = int(round(shift_x / dx))
    return np.roll(u0, shift_cells)


def solve_for_mu(config: LWConfig, mu: float) -> LWSimulation:
    x = config.x_grid()
    dt = config.dt(mu)
    nt = config.nt(mu)
    nu = config.a * dt / config.dx

    u0 = initial_profile(x)
    u = u0.copy()
    for _ in range(nt):
        u = step_lax_wendroff(u, nu)

    exact = exact_shift_periodic(u0, config.a * nt * dt, config.dx)
    return LWSimulation(mu=mu, nu=nu, x_grid=x, u0=exact, u_final=u)


def run_all(config: LWConfig) -> list[LWSimulation]:
    return [solve_for_mu(config, mu) for mu in config.mu_values]
