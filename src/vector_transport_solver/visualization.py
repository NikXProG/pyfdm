import matplotlib.pyplot as plt
import numpy as np

from .analysis import crossing_points, mid_level
from .solver import VectorTransportResult


def plot_dynamic_frames(result: VectorTransportResult, stride: int, show: bool = True) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    ax_v, ax_w = axes

    for n in range(0, result.t_grid.size, max(1, stride)):
        v = result.v_history[n]
        w = result.w_history[n]
        t = result.t_grid[n]

        lv = mid_level(v)
        lw = mid_level(w)
        xv = crossing_points(result.x_grid, v, lv)
        xw = crossing_points(result.x_grid, w, lw)

        ax_v.clear()
        ax_v.plot(result.x_grid, v, color="tab:blue", label=f"v(x,t), t={t:.2f}")
        ax_v.axhline(lv, color="tab:gray", linestyle="--", linewidth=1.0, label="mid-level")
        if xv.size > 0:
            ax_v.scatter(xv, np.full_like(xv, lv), color="tab:red", s=30, zorder=5, label="crossings")
        ax_v.set_title("Dynamic v(x,t)")
        ax_v.set_xlabel("x")
        ax_v.set_ylabel("v")
        ax_v.grid(True, alpha=0.35)
        ax_v.legend(loc="upper right")

        ax_w.clear()
        ax_w.plot(result.x_grid, w, color="tab:green", label=f"w(x,t), t={t:.2f}")
        ax_w.axhline(lw, color="tab:gray", linestyle="--", linewidth=1.0, label="mid-level")
        if xw.size > 0:
            ax_w.scatter(xw, np.full_like(xw, lw), color="tab:red", s=30, zorder=5, label="crossings")
        ax_w.set_title("Dynamic w(x,t)")
        ax_w.set_xlabel("x")
        ax_w.set_ylabel("w")
        ax_w.grid(True, alpha=0.35)
        ax_w.legend(loc="upper right")

        if show:
            plt.pause(0.001)

    plt.tight_layout()


def plot_3d_surfaces(result: VectorTransportResult, show: bool = True) -> None:
    x_mesh, t_mesh = np.meshgrid(result.x_grid, result.t_grid)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.plot_surface(x_mesh, t_mesh, result.v_history, cmap="viridis", edgecolor="none")
    ax1.set_title("Surface v(x,t)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_zlabel("v")

    ax2.plot_surface(x_mesh, t_mesh, result.w_history, cmap="plasma", edgecolor="none")
    ax2.set_title("Surface w(x,t)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("w")

    plt.tight_layout()
    if show:
        plt.show()
