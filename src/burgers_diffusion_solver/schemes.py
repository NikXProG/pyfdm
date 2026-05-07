import numpy as np


def ux_upwind(u: np.ndarray, dx: float) -> np.ndarray:
    ux = np.zeros_like(u)
    pos = u[1:-1] >= 0.0
    neg = ~pos

    ux_inner = np.zeros_like(u[1:-1])
    ux_inner[pos] = (u[1:-1][pos] - u[:-2][pos]) / dx
    ux_inner[neg] = (u[2:][neg] - u[1:-1][neg]) / dx
    ux[1:-1] = ux_inner
    return ux


def uxx_central(u: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    return out


def explicit_step(u: np.ndarray, epsilon: float, dt: float, dx: float) -> np.ndarray:
    ux = ux_upwind(u, dx)
    uxx = uxx_central(u, dx)
    return u - dt * u * ux + dt * epsilon * uxx
