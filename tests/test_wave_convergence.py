import numpy as np

from wave_solver_core import solve_wave_neumann
from wave_task13_solver.config import WaveTask13Config
from wave_task13_solver.physics import exact_solution, initial_displacement, initial_velocity


def _run_error(nx: int) -> float:
    config = WaveTask13Config(nx=nx, t_final=1.0, cfl=0.7)
    x = config.x_grid()
    t = config.t_grid()
    result = solve_wave_neumann(
        x=x,
        t=t,
        dx=config.dx,
        initial_displacement=initial_displacement,
        initial_velocity=initial_velocity,
        exact_solution=exact_solution,
    )
    return float(np.max(np.abs(result.u_num_final - result.u_exact_final)))


def test_wave_solver_has_second_order_like_convergence() -> None:
    e_coarse = _run_error(101)
    e_mid = _run_error(201)
    e_fine = _run_error(401)
    assert e_mid < 0.7 * e_coarse
    assert e_fine < 0.7 * e_mid
