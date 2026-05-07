from typing import Callable

import numpy as np

StepScheme = Callable[[np.ndarray, float], np.ndarray]


def step_upwind(u: np.ndarray, nu: float) -> np.ndarray:
    u_new = u.copy()
    u_new[1:] = u[1:] - nu * (u[1:] - u[:-1])
    return u_new


def step_lax_wendroff(u: np.ndarray, nu: float) -> np.ndarray:
    u_new = u.copy()
    u_new[1:-1] = (
        u[1:-1]
        - 0.5 * nu * (u[2:] - u[:-2])
        + 0.5 * (nu**2) * (u[2:] - 2.0 * u[1:-1] + u[:-2])
    )
    return u_new


def step_beam_warming(u: np.ndarray, nu: float) -> np.ndarray:
    u_new = u.copy()
    u_new[2:-1] = (
        u[2:-1]
        - 0.5 * nu * (3.0 * u[2:-1] - 4.0 * u[1:-2] + u[:-3])
        + 0.5 * (nu**2) * (u[2:-1] - 2.0 * u[1:-2] + u[:-3])
    )
    return u_new


def step_fromm(u: np.ndarray, nu: float) -> np.ndarray:
    u_new = u.copy()
    u_i = u[2:-1]
    u_im1 = u[1:-2]
    u_im2 = u[:-3]
    u_ip1 = u[3:]

    f1_lw = u_ip1 - u_im1
    f2_lw = u_ip1 - 2.0 * u_i + u_im1
    f1_bw = 3.0 * u_i - 4.0 * u_im1 + u_im2
    f2_bw = u_i - 2.0 * u_im1 + u_im2

    u_new[2:-1] = u_i - 0.25 * nu * (f1_lw + f1_bw) + 0.25 * (nu**2) * (f2_lw + f2_bw)
    return u_new


def scheme_registry() -> dict[str, StepScheme]:
    return {
        "Upwind": step_upwind,
        "Lax-Wendroff": step_lax_wendroff,
        "Beam-Warming": step_beam_warming,
        "Fromm": step_fromm,
    }
