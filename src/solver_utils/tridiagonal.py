import numpy as np


def solve_tridiagonal_system(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    n = diagonal.size
    if rhs.size != n:
        raise ValueError("RHS size must match diagonal size.")
    if lower.size != max(0, n - 1) or upper.size != max(0, n - 1):
        raise ValueError("Lower/upper diagonals must have size n-1.")
    if n == 0:
        return np.array([], dtype=float)

    a = lower.astype(float, copy=True)
    b = diagonal.astype(float, copy=True)
    c = upper.astype(float, copy=True)
    d = rhs.astype(float, copy=True)

    for i in range(1, n):
        if np.isclose(b[i - 1], 0.0):
            raise ValueError("Encountered zero pivot in tridiagonal solver.")
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    if np.isclose(b[-1], 0.0):
        raise ValueError("Encountered zero pivot in tridiagonal solver.")

    x = np.zeros_like(d)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        if np.isclose(b[i], 0.0):
            raise ValueError("Encountered zero pivot in tridiagonal solver.")
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x
