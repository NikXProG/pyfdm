import matplotlib.pyplot as plt
import numpy as np

from .config import WaveTask13Config
from .solver import WaveTask13Result


def plot_results(config: WaveTask13Config, result: WaveTask13Result, show: bool = True) -> None:
    x = result.x_grid
    t = result.t_grid
    u = result.u_num_history
    exact = result.u_exact_final
    num = result.u_num_final
    idx = result.max_error_indices

    sample_times = np.linspace(0.0, config.t_final, 5)
    sample_idx = [int(np.argmin(np.abs(t - ts))) for ts in sample_times]

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    for i in sample_idx:
        ax1.plot(x, u[i], linewidth=1.7, label=f"t={t[i]:.2f}")
    ax1.set_title("Задача 13: численное решение в разные моменты времени")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x,t)")
    ax1.grid(True, alpha=0.35)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, num, linewidth=2.0, label="приближенное")
    ax2.plot(x, exact, "--", linewidth=2.0, label="точное")
    ax2.scatter(
        x[idx],
        num[idx],
        color="purple",
        s=48,
        zorder=5,
        label="точки макс. отклонения",
    )
    ax2.set_title("Сравнение при t = T и точки максимальной ошибки")
    ax2.set_xlabel("x")
    ax2.set_ylabel("u(x,T)")
    ax2.grid(True, alpha=0.35)
    ax2.legend()

    plt.tight_layout()
    if show:
        plt.show()
