from typing import Callable

import numpy as np

UxScheme = Callable[[np.ndarray, float], np.ndarray]


def ux_backward(u: np.ndarray, dx: float) -> np.ndarray:
    ux = np.zeros_like(u)
    ux[1:-1] = (u[1:-1] - u[:-2]) / dx
    return ux


def ux_forward(u: np.ndarray, dx: float) -> np.ndarray:
    ux = np.zeros_like(u)
    ux[1:-1] = (u[2:] - u[1:-1]) / dx
    return ux


def ux_central(u: np.ndarray, dx: float) -> np.ndarray:
    ux = np.zeros_like(u)
    ux[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    return ux


def uxx_central(u: np.ndarray, dx: float) -> np.ndarray:
    uxx = np.zeros_like(u)
    uxx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    return uxx


def ux_schemes_registry() -> dict[str, UxScheme]:
    return {
        "Backward": ux_backward,
        "Forward": ux_forward,
        "Central": ux_central,
    }
