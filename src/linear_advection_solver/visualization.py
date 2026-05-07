import matplotlib.pyplot as plt
import numpy as np

from .solver import SimulationResult


def plot_comparison(results: list[SimulationResult], exact: np.ndarray, x_grid: np.ndarray) -> None:
    mask = (x_grid >= 15.0) & (x_grid <= 25.0)
    x_view = x_grid[mask]

    plt.figure(figsize=(12, 6))
    plt.plot(x_view, exact[mask], color="black", linewidth=2.2, label="Exact")

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for color, result in zip(palette, results):
        plt.plot(x_view, result.u_final[mask], color=color, linewidth=1.8, label=result.scheme_name)

    plt.title("Linear Advection at t=17 (window [15, 25])")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()
