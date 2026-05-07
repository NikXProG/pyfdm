import numpy as np


def amplification_factor(theta: np.ndarray, nu: float) -> np.ndarray:
    return 1.0 - 1j * nu * np.sin(theta) + (nu**2) * (np.cos(theta) - 1.0)


def max_amplification_abs(nu: float, n_theta: int = 4000) -> float:
    theta = np.linspace(-np.pi, np.pi, n_theta)
    g = amplification_factor(theta, nu)
    return float(np.max(np.abs(g)))


def stability_summary() -> str:
    return "\n".join(
        [
            "Спектральный критерий Неймана для схемы Лакса-Вендроффа:",
            "G(theta) = 1 - i*nu*sin(theta) + nu^2*(cos(theta)-1), nu=a*mu.",
            "|G(theta)| <= 1 для всех theta тогда и только тогда, когда |nu| <= 1.",
            "Следовательно при a>0 условие устойчивости: a*mu <= 1.",
        ]
    )
