import numpy as np
from solver_utils import solve_tridiagonal_system as tdma_solve


def tridiagonal_matrix(main: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    n = main.size
    out = np.zeros((n, n), dtype=float)
    out[np.arange(n), np.arange(n)] = main
    out[np.arange(1, n), np.arange(n - 1)] = low
    out[np.arange(n - 1), np.arange(1, n)] = high
    return out


def solve_tridiagonal(main: np.ndarray, low: np.ndarray, high: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return tdma_solve(low, main, high, rhs)
