import numpy as np

from .config import CoupledConfig
from .datasets import Dataset
from .linear_system import solve_tridiagonal


def solve_t_implicit(config: CoupledConfig, data: Dataset) -> np.ndarray:
    x = config.x_grid()
    t = config.t_grid()
    nx = x.size
    nt = t.size

    u = np.zeros((nt, nx), dtype=float)
    u[0] = data.t0(x)
    u[:, 0] = data.u1(t)
    u[:, -1] = data.u2(t)

    r = config.d0 * config.dt / (config.dx * config.dx)
    n_inner = nx - 2
    main = (1.0 + 2.0 * r) * np.ones(n_inner)
    low = -r * np.ones(n_inner - 1)
    high = -r * np.ones(n_inner - 1)

    for n in range(nt - 1):
        rhs = u[n, 1:-1].copy()
        rhs[0] += r * u[n + 1, 0]
        rhs[-1] += r * u[n + 1, -1]
        u[n + 1, 1:-1] = solve_tridiagonal(main, low, high, rhs)

    return u
