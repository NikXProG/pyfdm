import numpy as np


def initial_profile(x: np.ndarray) -> np.ndarray:
    return 3.0 * np.exp(-(x - 2.0) ** 2) + np.exp(-(x - 4.0) ** 2)


def apply_boundaries(u: np.ndarray) -> None:
    u[0] = 0.0
    u[-1] = 0.0
