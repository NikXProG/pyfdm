from dataclasses import dataclass

import numpy as np

from .config import WaveTask13Config
from .physics import exact_solution, initial_displacement, initial_velocity
from wave_solver_core import solve_wave_neumann


@dataclass
class WaveTask13Result:
    x_grid: np.ndarray
    t_grid: np.ndarray
    u_num_history: np.ndarray
    u_exact_final: np.ndarray
    u_num_final: np.ndarray
    max_error_indices: np.ndarray


def solve(config: WaveTask13Config) -> WaveTask13Result:
    x = config.x_grid()
    t = config.t_grid()
    wave = solve_wave_neumann(
        x=x,
        t=t,
        dx=config.dx,
        initial_displacement=initial_displacement,
        initial_velocity=initial_velocity,
        exact_solution=exact_solution,
    )

    return WaveTask13Result(
        x_grid=x,
        t_grid=t,
        u_num_history=wave.u_num_history,
        u_exact_final=wave.u_exact_final,
        u_num_final=wave.u_num_final,
        max_error_indices=wave.max_error_indices,
    )
