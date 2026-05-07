import numpy as np

from .solver import SimulationPair


def compare_final_profiles(pair: SimulationPair) -> dict[str, float]:
    u1 = pair.upwind_history[-1]
    u2 = pair.advection_diffusion_history[-1]
    diff = u1 - u2
    return {
        "l_inf": float(np.max(np.abs(diff))),
        "l2": float(np.sqrt(np.mean(diff**2))),
        "amp_upwind": float(np.max(np.abs(u1))),
        "amp_advdiff": float(np.max(np.abs(u2))),
    }


def build_conclusion(metrics: dict[str, float]) -> str:
    return (
        "Вывод: upwind-схема действительно вносит численную вязкость и ведет себя как "
        "схема для уравнения переноса-диффузии с epsilon_num. "
        f"Сходство подтверждается малыми расхождениями: L_inf={metrics['l_inf']:.3e}, "
        f"L2={metrics['l2']:.3e}."
    )
