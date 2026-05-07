import numpy as np

from .solver import SigmaRun


def max_point_at_final(run: SigmaRun) -> tuple[float, float]:
    y = run.u_history[-1]
    idx = int(np.argmax(y))
    return float(run.x_grid[idx]), float(y[idx])


def print_summary(runs: list[SigmaRun]) -> None:
    print("\nСводка по sigma-схеме при T=5:")
    for run in runs:
        x_max, u_max = max_point_at_final(run)
        print(f"sigma={run.sigma:.1f}: max u={u_max:.4f} at x={x_max:.4f}")
