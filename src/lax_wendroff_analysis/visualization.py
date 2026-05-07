import matplotlib.pyplot as plt

from .analysis import metrics
from .solver import LWSimulation


def plot_sims(sims: list[LWSimulation]) -> None:
    fig, axes = plt.subplots(1, len(sims), figsize=(5 * len(sims), 4), sharey=True)
    if len(sims) == 1:
        axes = [axes]

    for ax, sim in zip(axes, sims):
        m = metrics(sim)
        ax.plot(sim.x_grid, sim.u0, color="black", linewidth=1.8, label="exact")
        ax.plot(sim.x_grid, sim.u_final, color="tab:blue", linewidth=1.6, label="LW")
        ax.set_title(f"mu={sim.mu:.2f}, max|G|={m['g_max']:.3f}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.35)
    axes[0].set_ylabel("u(x,t)")
    axes[0].legend()
    fig.suptitle("Lax-Wendroff: stable and unstable Courant regimes")
    fig.tight_layout()
    plt.show()
