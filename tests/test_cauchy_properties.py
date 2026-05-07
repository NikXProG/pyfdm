import numpy as np

from cauchy_solver.config import SimulationConfig
from cauchy_solver.solver import solve


def test_cauchy_dirichlet_boundaries_stay_zero() -> None:
    result = solve(SimulationConfig(nx=101, nt=200, t_final=1.0))
    assert np.allclose(result.u_history[:, 0], 0.0)
    assert np.allclose(result.u_history[:, -1], 0.0)
