import numpy as np

from implicit_heat_solver.config import ImplicitHeatConfig
from implicit_heat_solver.solver import solve


def _restrict_to_coarse(fine_values: np.ndarray, stride: int) -> np.ndarray:
    return fine_values[::stride]


def test_implicit_heat_self_convergence_is_first_order_or_better() -> None:
    coarse = solve(ImplicitHeatConfig(x_min=-2.0, x_max=2.0, dx=0.2, t_final=0.5, dt=0.05))
    mid = solve(ImplicitHeatConfig(x_min=-2.0, x_max=2.0, dx=0.1, t_final=0.5, dt=0.025))
    fine = solve(ImplicitHeatConfig(x_min=-2.0, x_max=2.0, dx=0.05, t_final=0.5, dt=0.0125))

    coarse_final = coarse.u_history[-1]
    mid_final = _restrict_to_coarse(mid.u_history[-1], stride=2)
    fine_final = _restrict_to_coarse(fine.u_history[-1], stride=4)

    diff_coarse_mid = np.linalg.norm(coarse_final - mid_final, ord=np.inf)
    diff_mid_fine = np.linalg.norm(mid_final - fine_final, ord=np.inf)
    assert diff_mid_fine < diff_coarse_mid
    assert diff_coarse_mid / max(diff_mid_fine, 1e-14) > 1.5


def test_implicit_heat_respects_boundary_and_damps_amplitude() -> None:
    result = solve(ImplicitHeatConfig(x_min=-2.0, x_max=2.0, dx=0.1, t_final=0.5, dt=0.01))
    assert np.allclose(result.u_history[:, 0], 0.0)
    assert np.allclose(result.u_history[:, -1], 0.0)
    max_abs = np.max(np.abs(result.u_history), axis=1)
    assert np.all(max_abs[1:] <= max_abs[:-1] + 1e-10)
