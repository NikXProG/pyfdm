import numpy as np


def initial_profile(x: np.ndarray) -> np.ndarray:
    u = np.zeros_like(x)
    u[(x >= 1.0) & (x <= 2.0)] = 1.0
    u[(x >= 3.0) & (x <= 5.0)] = 2.0
    return u


def apply_dirichlet_boundaries(u: np.ndarray) -> None:
    u[0] = 0.0
    u[-1] = 0.0
