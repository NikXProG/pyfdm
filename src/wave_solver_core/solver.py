from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class WaveSolveResult:
    u_num_history: np.ndarray
    u_exact_final: np.ndarray
    u_num_final: np.ndarray
    max_error_indices: np.ndarray


def _laplacian_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    lap = np.zeros_like(u)
    dx2 = dx * dx
    lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx2
    lap[0] = 2.0 * (u[1] - u[0]) / dx2
    lap[-1] = 2.0 * (u[-2] - u[-1]) / dx2
    return lap


def solve_wave_neumann(
    x: np.ndarray,
    t: np.ndarray,
    dx: float,
    initial_displacement: Callable[[np.ndarray], np.ndarray],
    initial_velocity: Callable[[np.ndarray], np.ndarray],
    exact_solution: Callable[[np.ndarray, float], np.ndarray],
) -> WaveSolveResult:
    nt = t.size - 1
    dt = t[1] - t[0]
    dt2 = dt * dt

    u_num = np.zeros((nt + 1, x.size))
    u0 = initial_displacement(x)
    v0 = initial_velocity(x)
    u_num[0] = u0
    u_num[1] = u0 + dt * v0 + 0.5 * dt2 * _laplacian_neumann(u0, dx)

    for n in range(1, nt):
        lap = _laplacian_neumann(u_num[n], dx)
        u_num[n + 1] = 2.0 * u_num[n] - u_num[n - 1] + dt2 * lap

    u_num_final = u_num[-1]
    u_exact_final = exact_solution(x, t[-1])
    err = np.abs(u_num_final - u_exact_final)
    max_error = float(np.max(err))
    max_idx = np.where(np.isclose(err, max_error))[0]
    return WaveSolveResult(
        u_num_history=u_num,
        u_exact_final=u_exact_final,
        u_num_final=u_num_final,
        max_error_indices=max_idx,
    )
