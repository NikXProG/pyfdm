from .solver import RecalculationResult


def final_max_point(result: RecalculationResult) -> tuple[float, float]:
    y = result.u_history[-1]
    idx = int(y.argmax())
    return float(result.x_grid[idx]), float(y[idx])
