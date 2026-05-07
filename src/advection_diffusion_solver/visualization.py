import matplotlib.pyplot as plt

from .solver import SimulationResult


def plot_results(results: list[SimulationResult]) -> None:
    # 3 сценария x 3 схемы = 9 графиков
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for idx, result in enumerate(results):
        ax = axes_flat[idx]
        ax.plot(result.x_grid, result.u_history[0], color="gray", linestyle="--", linewidth=1.2, label="t=0")
        ax.plot(result.x_grid, result.u_history[len(result.t_grid) // 2], color="tab:blue", linewidth=1.5, label="t=T/2")
        ax.plot(result.x_grid, result.u_history[-1], color="tab:red", linewidth=2.0, label="t=T")
        ax.set_title(f"{result.scenario.name} | {result.ux_scheme_name}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.grid(True, alpha=0.35)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Advection-Diffusion: explicit Euler, comparison of u_x approximations", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
