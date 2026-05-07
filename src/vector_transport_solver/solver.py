from dataclasses import dataclass

import numpy as np

from .config import VectorTransportConfig
from .initial_conditions import build_initial_u
from .schemes import upwind_scalar_step
from .transform import from_characteristic, to_characteristic


@dataclass
class VectorTransportResult:
    x_grid: np.ndarray
    t_grid: np.ndarray
    v_history: np.ndarray
    w_history: np.ndarray


def solve(config: VectorTransportConfig) -> VectorTransportResult:
    x = config.x_grid()
    dt = config.dt()
    nt = config.nt()
    t_grid = np.linspace(0.0, nt * dt, nt + 1)

    lambdas, vecs, inv_vecs = config.eigen_system()
    u = build_initial_u(x)
    z = to_characteristic(u, inv_vecs)

    v_history = np.zeros((nt + 1, x.size))
    w_history = np.zeros((nt + 1, x.size))
    v_history[0] = u[0]
    w_history[0] = u[1]

    for n in range(1, nt + 1):
        for k in range(2):
            z[k] = upwind_scalar_step(z[k], lambdas[k], dt, config.dx)

        u = from_characteristic(z, vecs)
        v_history[n] = u[0]
        w_history[n] = u[1]

    return VectorTransportResult(x_grid=x, t_grid=t_grid, v_history=v_history, w_history=w_history)
