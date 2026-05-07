import numpy as np


def box_pulse_initial(x: np.ndarray, left: float = 4.9, right: float = 5.1) -> np.ndarray:
    u0 = np.zeros_like(x)
    mask = (x >= left) & (x <= right)
    u0[mask] = 1.0
    return u0


def apply_dirichlet_boundaries(u: np.ndarray, left_value: float = 0.0, right_value: float = 0.0) -> None:
    u[0] = left_value
    u[-1] = right_value
