import numpy as np


def step_lax_wendroff(u: np.ndarray, nu: float) -> np.ndarray:
    u_right = np.roll(u, -1)
    u_left = np.roll(u, 1)
    return u - 0.5 * nu * (u_right - u_left) + 0.5 * (nu**2) * (u_right - 2.0 * u + u_left)
