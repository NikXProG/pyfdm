from dataclasses import dataclass

import numpy as np

from .solver import SimulationResult


@dataclass(frozen=True)
class Observation:
    max_abs_initial: float
    max_abs_final: float
    growth_ratio: float
    min_final: float
    max_final: float
    total_variation_final: float
    status: str
    comment: str


def build_observation(result: SimulationResult) -> Observation:
    initial = result.u_history[0]
    final = result.u_history[-1]

    max_abs_initial = float(np.max(np.abs(initial)))
    max_abs_final = float(np.max(np.abs(final)))
    growth_ratio = max_abs_final / max(max_abs_initial, 1e-12)
    min_final = float(np.min(final))
    max_final = float(np.max(final))
    total_variation_final = float(np.sum(np.abs(np.diff(final))))

    if growth_ratio > 20.0 or not np.isfinite(growth_ratio):
        status = "unstable"
        comment = "Наблюдается сильный рост амплитуды (численная неустойчивость)."
    elif min_final < -1e-3:
        status = "oscillatory"
        comment = "Есть заметные осцилляции и отрицательные выбросы."
    else:
        status = "stable"
        comment = "Решение ведет себя физично и остается ограниченным."

    return Observation(
        max_abs_initial=max_abs_initial,
        max_abs_final=max_abs_final,
        growth_ratio=growth_ratio,
        min_final=min_final,
        max_final=max_final,
        total_variation_final=total_variation_final,
        status=status,
        comment=comment,
    )


def print_observations(results: list[SimulationResult]) -> None:
    print("\nНаблюдения по схемам (задача 2):")
    print("-" * 78)
    for result in results:
        obs = build_observation(result)
        print(
            f"{result.scenario.name:20} | {result.ux_scheme_name:8} | "
            f"status={obs.status:10} | growth={obs.growth_ratio:10.3e} | "
            f"min={obs.min_final:10.3e} | max={obs.max_final:10.3e}"
        )
        print(f"  -> {obs.comment}")
