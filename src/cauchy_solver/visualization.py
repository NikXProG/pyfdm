import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .solver import SimulationResult


def _snapshot_indices(t_grid: np.ndarray, requested_times: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(t_grid - ts))) for ts in requested_times], dtype=int)


def plot_results(config: SimulationConfig, result: SimulationResult, show: bool = True) -> None:
    x_grid = result.x_grid
    t_grid = result.t_grid
    u_history = result.u_history
    plot_times = np.linspace(0.0, config.t_final, 6)
    idx = _snapshot_indices(t_grid, plot_times)

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 1, 1)
    for i in idx:
        ax1.plot(x_grid, u_history[i], linewidth=1.8, label=f"t={t_grid[i]:.2f}")
    ax1.set_title("Решение u(x,t) для разных моментов времени")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x, t)")
    ax1.grid(True, alpha=0.35)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2, projection="3d")
    x_mesh, t_mesh = np.meshgrid(x_grid, t_grid)
    ax2.plot_surface(x_mesh, t_mesh, u_history, cmap="viridis", edgecolor="none")
    ax2.set_title("Трехмерная поверхность u(x,t)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("u")

    plt.tight_layout()
    if show:
        plt.show()
