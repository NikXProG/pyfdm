from dataclasses import dataclass

import numpy as np

from .config import ImplicitHeatConfig
from .initial_conditions import initial_profile
from .linear_system import build_implicit_matrix, solve_tridiagonal_system


@dataclass
class HeatResult:
    x_grid: np.ndarray
    t_grid: np.ndarray
    u_history: np.ndarray
    snapshot_indices: np.ndarray


def solve(config: ImplicitHeatConfig) -> HeatResult:
    x = config.x_grid()
    t = config.t_grid()
    nx = x.size
    nt = t.size

    r = config.kappa * config.dt / (config.dx**2)
    n_inner = nx - 2
    a = build_implicit_matrix(n_inner, r)

    u = initial_profile(x)
    u[0] = 0.0
    u[-1] = 0.0

    u_history = np.zeros((nt, nx))
    u_history[0] = u.copy()

    for n in range(1, nt):
        rhs = u[1:-1].copy()
        inner = solve_tridiagonal_system(a, rhs)
        u_new = np.zeros_like(u)
        u_new[1:-1] = inner
        u = u_new
        u_history[n] = u.copy()

    snap_idx = []
    for ts in config.snapshot_times():
        snap_idx.append(int(np.argmin(np.abs(t - ts))))

    return HeatResult(x_grid=x, t_grid=t, u_history=u_history, snapshot_indices=np.array(snap_idx, dtype=int))
