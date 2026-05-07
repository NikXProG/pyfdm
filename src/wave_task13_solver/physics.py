import numpy as np

N_MODES = 300
N_QUAD = 4001


def initial_displacement(x: np.ndarray) -> np.ndarray:
    return 1.0 + np.cos(3.0 * np.pi * x)


def initial_velocity(x: np.ndarray) -> np.ndarray:
    return np.cos(2.0 * np.pi * x)


def _build_neumann_series_coeffs() -> tuple[float, float, np.ndarray, np.ndarray]:
    xq = np.linspace(0.0, 1.0, N_QUAD)
    u0 = initial_displacement(xq)
    v0 = initial_velocity(xq)
    a0 = float(np.trapezoid(u0, xq))
    b0 = float(np.trapezoid(v0, xq))

    n = np.arange(1, N_MODES + 1, dtype=float)
    basis = np.cos(np.pi * np.outer(xq, n))
    a = 2.0 * np.trapezoid(u0[:, None] * basis, xq, axis=0)
    b = 2.0 * np.trapezoid(v0[:, None] * basis, xq, axis=0)
    return a0, b0, a, b


A0, B0, A_COEFF, B_COEFF = _build_neumann_series_coeffs()


def exact_solution(x: np.ndarray, t: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = np.arange(1, N_MODES + 1, dtype=float)
    cos_x = np.cos(np.pi * np.outer(x, n))
    time_part = A_COEFF * np.cos(np.pi * n * t) + (B_COEFF / (np.pi * n)) * np.sin(np.pi * n * t)
    return A0 + B0 * t + cos_x @ time_part
