import numpy as np

from .initial_conditions import initial_profile


def exact_solution(x: np.ndarray, t: float, a: float) -> np.ndarray:
    shifted = x - a * t
    u = initial_profile(shifted)
    u[shifted < 0.0] = 0.0
    return u
