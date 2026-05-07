import matplotlib.pyplot as plt
import numpy as np

from .analysis import maxima_points
from .solver import HeatResult


def plot_snapshots_with_maxima(result: HeatResult) -> None:
    idx = result.snapshot_indices
    u_snaps = result.u_history[idx]
    t_snaps = result.t_grid[idx]
    max_pts = maxima_points(result.x_grid, u_snaps)

    plt.figure(figsize=(12, 6))
    for y, ts in zip(u_snaps, t_snaps):
        plt.plot(result.x_grid, y, linewidth=1.7, label=f"t={ts:.1f}")

    for (x_max, u_max), ts in zip(max_pts, t_snaps):
        plt.scatter(x_max, u_max, color="red", s=30, zorder=5)
        plt.text(x_max, u_max, f" max@t={ts:.1f}", fontsize=8)

    plt.title("Implicit scheme snapshots for 4u_t = u_xx")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=2)
    plt.tight_layout()


def plot_surface(result: HeatResult) -> None:
    x_mesh, t_mesh = np.meshgrid(result.x_grid, result.t_grid)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(x_mesh, t_mesh, result.u_history, cmap="viridis", edgecolor="none")
    ax.set_title("Surface u(x,t) for implicit scheme")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    plt.tight_layout()
    plt.show()
