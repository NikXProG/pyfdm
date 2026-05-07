import numpy as np


def step_upwind(u: np.ndarray, a: float, dt: float, dx: float) -> np.ndarray:
    nu = a * dt / dx
    u_new = u.copy()
    u_new[1:] = u[1:] - nu * (u[1:] - u[:-1])
    return u_new


def step_advection_diffusion(
    u: np.ndarray,
    a: float,
    epsilon: float,
    dt: float,
    dx: float,
) -> np.ndarray:
    u_new = u.copy()
    ux_upwind = (u[1:-1] - u[:-2]) / dx
    uxx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    u_new[1:-1] = u[1:-1] + dt * (-a * ux_upwind + epsilon * uxx)
    return u_new
