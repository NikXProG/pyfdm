import numpy as np


def crossing_points(x: np.ndarray, y: np.ndarray, level: float) -> np.ndarray:
    diff = y - level
    idx = np.where(diff[:-1] * diff[1:] <= 0.0)[0]
    points = []
    for i in idx:
        x1, x2 = x[i], x[i + 1]
        y1, y2 = diff[i], diff[i + 1]
        if np.isclose(y2, y1):
            points.append(0.5 * (x1 + x2))
        else:
            alpha = -y1 / (y2 - y1)
            points.append(x1 + alpha * (x2 - x1))
    return np.array(points)


def mid_level(y: np.ndarray) -> float:
    return 0.5 * (float(np.min(y)) + float(np.max(y)))
