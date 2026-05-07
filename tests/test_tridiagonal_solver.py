import numpy as np

from solver_utils import solve_tridiagonal_system


def test_tdma_matches_dense_solution() -> None:
    diagonal = np.array([4.0, 4.0, 4.0, 4.0])
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([-1.0, -1.0, -1.0])
    rhs = np.array([1.0, 2.0, 2.0, 1.0])

    dense = np.diag(diagonal) + np.diag(lower, k=-1) + np.diag(upper, k=1)
    expected = np.linalg.solve(dense, rhs)
    actual = solve_tridiagonal_system(lower, diagonal, upper, rhs)
    assert np.allclose(actual, expected)
