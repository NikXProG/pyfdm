import matplotlib.pyplot as plt
import numpy as np

from .analysis import final_max_point
from .solver import RecalculationResult


def plot_final_with_max(result: RecalculationResult) -> None:
    y = result.u_history[-1]
    x_max, u_max = final_max_point(result)

    plt.figure(figsize=(10, 5))
    plt.plot(result.x_grid, y, color="tab:blue", linewidth=2.0, label="u(x,T=7)")
    plt.scatter(x_max, u_max, color="red", s=45, zorder=5, label="maximum")
    plt.text(x_max, u_max, f" max=({x_max:.2f}, {u_max:.3f})", fontsize=9)
    plt.title("Recalculation scheme: final profile at T=7")
    plt.xlabel("x")
    plt.ylabel("u(x,7)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()


def plot_surface(result: RecalculationResult) -> None:
    x_mesh, t_mesh = np.meshgrid(result.x_grid, result.t_grid)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(x_mesh, t_mesh, result.u_history, cmap="viridis", edgecolor="none")
    ax.set_title("Surface u(x,t) for recalculation scheme")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    plt.tight_layout()
    plt.show()
