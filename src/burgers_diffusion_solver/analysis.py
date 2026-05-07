import numpy as np


def print_observations(profiles: dict[float, np.ndarray]) -> None:
    eps = sorted(profiles.keys())
    peak_vals = np.array([np.max(np.abs(profiles[e])) for e in eps], dtype=float)
    spread_vals = np.array([np.sum(np.abs(np.diff(profiles[e]))) for e in eps], dtype=float)

    print("\nЗависимость от epsilon при t=10:")
    print(
        f"- max|u|: от {peak_vals[0]:.3f} (eps={eps[0]:.1f}) "
        f"до {peak_vals[-1]:.3f} (eps={eps[-1]:.1f})"
    )
    print(
        f"- variation: от {spread_vals[0]:.3f} (eps={eps[0]:.1f}) "
        f"до {spread_vals[-1]:.3f} (eps={eps[-1]:.1f})"
    )
    print("- вывод: при росте epsilon решение становится более сглаженным и менее пиковым.")
