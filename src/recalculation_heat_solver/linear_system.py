import numpy as np
from solver_utils import solve_tridiagonal_system as tdma_solve


def build_half_step_matrix(n_inner: int, dt: float, dx: float) -> np.ndarray:
    r_half = (dt / 2.0) / (dx * dx)
    main = (1.0 + 2.0 * r_half) * np.ones(n_inner)
    off = -r_half * np.ones(n_inner - 1)
    return np.diag(main) + np.diag(off, 1) + np.diag(off, -1)


def solve_linear(a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lower = np.diag(a, k=-1)
    diagonal = np.diag(a, k=0)
    upper = np.diag(a, k=1)
    return tdma_solve(lower, diagonal, upper, rhs)
