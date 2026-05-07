import numpy as np


def maxima_points(x: np.ndarray, u_snapshots: np.ndarray) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for y in u_snapshots:
        idx = int(np.argmax(y))
        points.append((float(x[idx]), float(y[idx])))
    return points
