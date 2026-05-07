import numpy as np


def initial_profile(x: np.ndarray) -> np.ndarray:
    u = np.zeros_like(x)
    u[np.abs(x) <= 1.0] = 1.0
    return u


def apply_boundaries(u: np.ndarray) -> None:
    u[0] = 0.0
    u[-1] = 0.0
