import numpy as np
from solver_utils import solve_tridiagonal_system as tdma_solve


def build_implicit_matrix(n_inner: int, r: float) -> np.ndarray:
    main = (1.0 + 2.0 * r) * np.ones(n_inner)
    off = -r * np.ones(n_inner - 1)
    a = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)
    return a


def solve_tridiagonal_system(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lower = np.diag(matrix, k=-1)
    diagonal = np.diag(matrix, k=0)
    upper = np.diag(matrix, k=1)
    return tdma_solve(lower, diagonal, upper, rhs)
