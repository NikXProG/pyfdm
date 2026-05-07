import numpy as np


def initial_profile(x: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.exp(2.0 * x - x**2)
