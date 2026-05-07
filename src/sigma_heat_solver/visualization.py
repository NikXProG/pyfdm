import matplotlib.pyplot as plt
import numpy as np

from .analysis import max_point_at_final
from .solver import SigmaRun


def plot_final_curves_with_maxima(runs: list[SigmaRun]) -> None:
    plt.figure(figsize=(12, 6))
    for run in runs:
        y = run.u_history[-1]
        plt.plot(run.x_grid, y, linewidth=1.8, label=f"sigma={run.sigma:.1f}")
        x_max, u_max = max_point_at_final(run)
        plt.scatter(x_max, u_max, s=40, zorder=5)
        plt.text(x_max, u_max, f" max({x_max:.2f},{u_max:.2f})", fontsize=8)

    plt.title("Sigma-scheme solutions at T=5")
    plt.xlabel("x")
    plt.ylabel("u(x,5)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()


def plot_surface_for_run(run: SigmaRun) -> None:
    x_mesh, t_mesh = np.meshgrid(run.x_grid, run.t_grid)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(x_mesh, t_mesh, run.u_history, cmap="viridis", edgecolor="none")
    ax.set_title(f"Surface u(x,t), sigma={run.sigma:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    plt.tight_layout()
    plt.show()
