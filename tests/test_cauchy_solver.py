import numpy as np

from cauchy_solver.config import SimulationConfig
from cauchy_solver import solver as cauchy_solver


def test_cauchy_uses_initial_velocity_on_first_step() -> None:
    original = cauchy_solver.initial_velocity
    try:
        cauchy_solver.initial_velocity = lambda x: np.ones_like(x)
        config = SimulationConfig(nt=3)
        result = cauchy_solver.solve(config)
    finally:
        cauchy_solver.initial_velocity = original

    u_first = result.u_history[1, 1:-1]
    u0 = result.u_history[0, 1:-1]
    assert np.linalg.norm(u_first - u0) > 1e-8
