import matplotlib.pyplot as plt
import numpy as np

from .solver import BurgersResult


def plot_surface(result: BurgersResult) -> None:
    x_mesh, t_mesh = np.meshgrid(result.x_grid, result.t_grid)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(x_mesh, t_mesh, result.u_history, cmap="viridis", edgecolor="none")
    ax.set_title("u(x,t) for u_t + u u_x = eps u_xx")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    plt.tight_layout()


def plot_epsilon_sweep(x: np.ndarray, profiles: dict[float, np.ndarray]) -> None:
    plt.figure(figsize=(12, 6))
    for eps, u in profiles.items():
        plt.plot(x, u, linewidth=1.2, label=f"eps={eps:.1f}")
    plt.title("u(x, t=10) for epsilon = 0.1..2.0")
    plt.xlabel("x")
    plt.ylabel("u(x,10)")
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()
