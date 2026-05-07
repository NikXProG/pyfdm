import numpy as np

from .solver import LWSimulation
from .von_neumann import max_amplification_abs


def metrics(sim: LWSimulation) -> dict[str, float]:
    diff = sim.u_final - sim.u0
    return {
        "l_inf": float(np.max(np.abs(diff))),
        "l2": float(np.sqrt(np.mean(diff**2))),
        "amp_ratio": float(np.max(np.abs(sim.u_final)) / max(np.max(np.abs(sim.u0)), 1e-12)),
        "g_max": max_amplification_abs(sim.nu),
    }


def print_results(sims: list[LWSimulation]) -> None:
    print("\nЧисленная проверка устойчивости Лакса-Вендроффа:")
    print("-" * 86)
    for sim in sims:
        m = metrics(sim)
        state = "stable" if m["g_max"] <= 1.000001 else "unstable"
        print(
            f"mu={sim.mu:4.2f}, nu={sim.nu:4.2f} | {state:8} | "
            f"max|G|={m['g_max']:.4f} | L_inf={m['l_inf']:.3e} | L2={m['l2']:.3e}"
        )
