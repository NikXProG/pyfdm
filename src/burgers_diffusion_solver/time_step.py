import numpy as np


def safe_dt(u: np.ndarray, epsilon: float, dx: float, cfl_conv: float, cfl_diff: float) -> float:
    umax = float(np.max(np.abs(u)))
    dt_conv = np.inf if umax < 1e-12 else cfl_conv * dx / umax
    dt_diff = np.inf if epsilon < 1e-12 else cfl_diff * dx * dx / epsilon
    return float(min(dt_conv, dt_diff))
