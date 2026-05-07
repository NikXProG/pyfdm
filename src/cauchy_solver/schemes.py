import numpy as np


def build_spatial_operator_coeffs(
    x_grid: np.ndarray,
    diffusion_coeff: np.ndarray,
    reaction_coeff: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_inner = x_grid.size - 2
    dx2 = dx * dx

    a_left = 0.5 * (diffusion_coeff[1:-1] + diffusion_coeff[:-2]) / dx2
    a_right = 0.5 * (diffusion_coeff[1:-1] + diffusion_coeff[2:]) / dx2

    lower = a_left[1:].copy()
    upper = a_right[:-1].copy()
    diagonal = -(a_left + a_right) - reaction_coeff[1:-1]

    if diagonal.size != n_inner:
        raise ValueError("Inconsistent grid sizes for operator assembly.")

    return lower, diagonal, upper


