import time

import numpy as np

from solver_utils import solve_tridiagonal_system


def _dense_solve(lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    matrix = np.diag(diagonal) + np.diag(lower, -1) + np.diag(upper, 1)
    return np.linalg.solve(matrix, rhs)


def run_benchmark(n: int = 600, repeats: int = 8) -> str:
    rng = np.random.default_rng(42)
    lower = -rng.uniform(0.1, 0.5, size=n - 1)
    upper = -rng.uniform(0.1, 0.5, size=n - 1)
    diagonal = rng.uniform(2.5, 3.5, size=n)
    rhs = rng.normal(size=n)

    t0 = time.perf_counter()
    for _ in range(repeats):
        _dense_solve(lower, diagonal, upper, rhs)
    dense_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(repeats):
        solve_tridiagonal_system(lower, diagonal, upper, rhs)
    tdma_elapsed = time.perf_counter() - t1

    speedup = dense_elapsed / tdma_elapsed if tdma_elapsed > 0 else float("inf")
    return (
        f"n={n}, repeats={repeats}\n"
        f"dense_elapsed={dense_elapsed:.6f}s\n"
        f"tdma_elapsed={tdma_elapsed:.6f}s\n"
        f"speedup={speedup:.2f}x\n"
    )


if __name__ == "__main__":
    print(run_benchmark())
