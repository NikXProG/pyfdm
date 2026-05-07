import numpy as np


def compare_part_a(t_impl: np.ndarray, t_fourier: np.ndarray) -> dict[str, float]:
    diff = t_impl - t_fourier
    return {
        "linf": float(np.max(np.abs(diff))),
        "l2": float(np.sqrt(np.mean(diff**2))),
    }


def final_max(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    i = int(np.argmax(y))
    return float(x[i]), float(y[i])
