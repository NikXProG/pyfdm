import numpy as np

from .config import CoupledConfig
from .datasets import Dataset
from .linear_system import solve_tridiagonal


def diffusion_from_t(t_slice: np.ndarray, config: CoupledConfig) -> np.ndarray:
    temp = np.maximum(t_slice, 1e-6)
    return config.d1 * np.exp(-config.delta_e1 / (config.k * temp))


def solve_phi_implicit(config: CoupledConfig, data: Dataset, t_field: np.ndarray) -> np.ndarray:
    x = config.x_grid()
    t = config.t_grid()
    nx = x.size
    nt = t.size

    phi = np.zeros((nt, nx), dtype=float)
    phi[0] = data.f(x)

    dx2 = config.dx * config.dx
    dt = config.dt

    for n in range(nt - 1):
        d = diffusion_from_t(t_field[n + 1], config)
        d_face_p = 0.5 * (d[1:-1] + d[2:])
        d_face_m = 0.5 * (d[1:-1] + d[:-2])

        low = -(dt / dx2) * d_face_m[1:]
        high = -(dt / dx2) * d_face_p[:-1]
        main = 1.0 + (dt / dx2) * (d_face_p + d_face_m)
        rhs = phi[n, 1:-1].copy()

        # Neumann BC: phi_x=0 at both ends via ghost points phi_0=phi_1, phi_N=phi_{N-1}.
        main[0] -= (dt / dx2) * d_face_m[0]
        main[-1] -= (dt / dx2) * d_face_p[-1]

        phi[n + 1, 1:-1] = solve_tridiagonal(main, low, high, rhs)
        phi[n + 1, 0] = phi[n + 1, 1]
        phi[n + 1, -1] = phi[n + 1, -2]

    return phi
