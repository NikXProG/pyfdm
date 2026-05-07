import numpy as np


def upwind_scalar_step(z: np.ndarray, speed: float, dt: float, dx: float) -> np.ndarray:
    nu = speed * dt / dx
    z_new = z.copy()

    if speed >= 0.0:
        z_new[1:] = z[1:] - nu * (z[1:] - z[:-1])
        z_new[0] = z_new[1]  # нулевая производная на левой границе
    else:
        z_new[:-1] = z[:-1] - nu * (z[1:] - z[:-1])
        z_new[-1] = z_new[-2]  # нулевая производная на правой границе

    return z_new
