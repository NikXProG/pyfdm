from dataclasses import dataclass

import numpy as np

from .solver import SimulationResult


@dataclass(frozen=True)
class QualityMetrics:
    l_inf: float
    l2: float
    overshoot: float
    undershoot: float


def evaluate(result: SimulationResult, exact: np.ndarray, x_grid: np.ndarray) -> QualityMetrics:
    mask = (x_grid >= 15.0) & (x_grid <= 25.0)
    diff = result.u_final[mask] - exact[mask]

    l_inf = float(np.max(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff**2)))
    overshoot = float(max(0.0, np.max(result.u_final[mask]) - np.max(exact[mask])))
    undershoot = float(max(0.0, np.min(exact[mask]) - np.min(result.u_final[mask])))

    return QualityMetrics(l_inf=l_inf, l2=l2, overshoot=overshoot, undershoot=undershoot)


def print_observations(results: list[SimulationResult], exact: np.ndarray, x_grid: np.ndarray) -> None:
    scored = []
    for result in results:
        metrics = evaluate(result, exact, x_grid)
        scored.append((result.scheme_name, metrics))

    scored.sort(key=lambda item: (item[1].l_inf, item[1].l2))

    print("\nСравнение схем на интервале [15, 25], t=17:")
    print("-" * 86)
    for name, m in scored:
        print(
            f"{name:14} | L_inf={m.l_inf:10.3e} | L2={m.l2:10.3e} | "
            f"overshoot={m.overshoot:9.2e} | undershoot={m.undershoot:9.2e}"
        )

    best_name = scored[0][0]
    print(f"\nВывод: на выбранной сетке лучшая точность у схемы: {best_name}.")
