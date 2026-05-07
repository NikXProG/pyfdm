import matplotlib.pyplot as plt
import numpy as np

from .solver import SimulationPair


def plot_comparison(pair: SimulationPair) -> None:
    x = pair.x_grid
    u0 = pair.upwind_history[0]
    u_upwind = pair.upwind_history[-1]
    u_advdiff = pair.advection_diffusion_history[-1]
    abs_diff = np.abs(u_upwind - u_advdiff)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(x, u0, color="gray", linestyle="--", linewidth=1.5, label="initial")
    ax1.plot(x, u_upwind, color="tab:blue", linewidth=2.0, label="upwind")
    ax1.plot(x, u_advdiff, color="tab:orange", linewidth=2.0, label="advection-diffusion")
    ax1.set_title("Numerical viscosity interpretation")
    ax1.set_ylabel("u(x, t)")
    ax1.grid(True, alpha=0.35)
    ax1.legend()

    ax2.plot(x, abs_diff, color="tab:red", linewidth=1.8, label="|upwind - advection-diffusion|")
    ax2.set_xlabel("x")
    ax2.set_ylabel("absolute difference")
    ax2.grid(True, alpha=0.35)
    ax2.legend()

    plt.tight_layout()
    plt.show()
