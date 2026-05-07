import numpy as np

from .linear_system import build_half_step_matrix, solve_linear


def half_step_implicit(u_n: np.ndarray, dt: float, dx: float) -> np.ndarray:
    n_inner = u_n.size - 2
    a = build_half_step_matrix(n_inner, dt, dx)
    rhs = u_n[1:-1].copy()
    inner = solve_linear(a, rhs)
    u_half = np.zeros_like(u_n)
    u_half[1:-1] = inner
    return u_half


def full_step_recalculation(u_n: np.ndarray, u_half: np.ndarray, dt: float, dx: float) -> np.ndarray:
    u_np1 = np.zeros_like(u_n)
    lap_half = (u_half[2:] - 2.0 * u_half[1:-1] + u_half[:-2]) / (dx * dx)
    u_np1[1:-1] = u_n[1:-1] + dt * lap_half
    return u_np1
