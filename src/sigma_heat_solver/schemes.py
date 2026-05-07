import numpy as np

from .linear_system import build_matrix, solve_system


def laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    return out


def explicit_step(u: np.ndarray, r: float) -> np.ndarray:
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
    return u_new


def sigma_step(u: np.ndarray, sigma: float, r: float, dx: float) -> np.ndarray:
    if np.isclose(sigma, 0.0):
        return explicit_step(u, r)

    n_inner = u.size - 2
    a = build_matrix(n_inner, sigma, r)
    rhs = u[1:-1] + (1.0 - sigma) * r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
    inner = solve_system(a, rhs)
    u_new = np.zeros_like(u)
    u_new[1:-1] = inner
    return u_new
