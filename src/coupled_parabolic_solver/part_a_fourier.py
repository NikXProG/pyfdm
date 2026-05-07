import numpy as np

from .config import CoupledConfig
from .datasets import Dataset


def _lifting(x: np.ndarray, t: float, data: Dataset) -> np.ndarray:
    return (1.0 - x) * data.u1(np.array([t]))[0] + x * data.u2(np.array([t]))[0]


def _forcing_coeff(n: int, x: np.ndarray, t: float, data: Dataset) -> float:
    g_t = (1.0 - x) * data.u1_dt(np.array([t]))[0] + x * data.u2_dt(np.array([t]))[0]
    basis = np.sin(n * np.pi * x)
    return -2.0 * np.trapezoid(g_t * basis, x)


def _initial_coeff(n: int, x: np.ndarray, data: Dataset) -> float:
    w0 = data.t0(x) - _lifting(x, 0.0, data)
    basis = np.sin(n * np.pi * x)
    return 2.0 * np.trapezoid(w0 * basis, x)


def _select_n_terms(config: CoupledConfig, data: Dataset) -> int:
    x = config.x_grid()
    tol = config.tol_fourier
    consecutive_below_tol = 0
    window = 5
    for n in range(5, 300):
        cn = abs(_initial_coeff(n, x, data))
        fn = abs(_forcing_coeff(n, x, 0.0, data))
        if cn + fn < tol:
            consecutive_below_tol += 1
            if consecutive_below_tol >= window:
                return n
        else:
            consecutive_below_tol = 0
    return 300


def solve_t_fourier(config: CoupledConfig, data: Dataset) -> tuple[np.ndarray, int]:
    x = config.x_grid()
    t = config.t_grid()
    nt = t.size
    nx = x.size
    n_terms = _select_n_terms(config, data)

    a = np.zeros((n_terms + 1,), dtype=float)
    for n in range(1, n_terms + 1):
        a[n] = _initial_coeff(n, x, data)

    out = np.zeros((nt, nx), dtype=float)
    out[0] = data.t0(x)

    for k in range(nt - 1):
        tk = t[k]
        tnext = t[k + 1]
        dt = tnext - tk
        w_next = np.zeros_like(x)
        for n in range(1, n_terms + 1):
            lam = config.d0 * (n * np.pi) ** 2
            b = _forcing_coeff(n, x, tk, data)
            a[n] = (a[n] + dt * b) / (1.0 + dt * lam)
            w_next += a[n] * np.sin(n * np.pi * x)
        out[k + 1] = w_next + _lifting(x, tnext, data)

    return out, n_terms
