import numpy as np
from solver_utils import solve_tridiagonal_system as tdma_solve


def build_matrix(n_inner: int, sigma: float, r: float) -> np.ndarray:
    main = (1.0 + 2.0 * sigma * r) * np.ones(n_inner)
    off = (-sigma * r) * np.ones(n_inner - 1)
    return np.diag(main) + np.diag(off, 1) + np.diag(off, -1)


def solve_system(a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lower = np.diag(a, k=-1)
    diagonal = np.diag(a, k=0)
    upper = np.diag(a, k=1)
    return tdma_solve(lower, diagonal, upper, rhs)
