import matplotlib.pyplot as plt
import numpy as np


def plot_case(
    case_id: str,
    x: np.ndarray,
    t: np.ndarray,
    t_impl: np.ndarray,
    t_fourier: np.ndarray,
    phi: np.ndarray,
    show: bool = True,
) -> None:
    fig = plt.figure(figsize=(14, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    ax1.plot(x, t_impl[0], label="T implicit t=0")
    ax1.plot(x, t_impl[-1], label="T implicit t=end")
    ax1.plot(x, t_fourier[-1], "--", label="T fourier t=end")
    ax1.set_title(f"Case {case_id}: T profiles")
    ax1.grid(True, alpha=0.35)
    ax1.legend(fontsize=8)

    ax2.plot(x, phi[0], label="phi t=0")
    ax2.plot(x, phi[-1], label="phi t=end")
    ax2.set_title(f"Case {case_id}: phi profiles")
    ax2.grid(True, alpha=0.35)
    ax2.legend(fontsize=8)

    xm, tm = np.meshgrid(x, t)
    ax3.plot_surface(xm, tm, t_impl, cmap="viridis", edgecolor="none")
    ax3.set_title("T(x,t) implicit")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    ax3.set_zlabel("T")

    ax4.plot_surface(xm, tm, phi, cmap="plasma", edgecolor="none")
    ax4.set_title("phi(x,t)")
    ax4.set_xlabel("x")
    ax4.set_ylabel("t")
    ax4.set_zlabel("phi")

    fig.suptitle(f"Task 15 case {case_id}")
    plt.tight_layout()
    if show:
        plt.show()
