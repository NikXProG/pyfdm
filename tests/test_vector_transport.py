import pytest

from vector_transport_solver.config import VectorTransportConfig
from vector_transport_solver.solver import solve


def test_vector_transport_time_grid_matches_effective_dt() -> None:
    config = VectorTransportConfig(t_final=1.0)
    result = solve(config)
    dt = config.dt()
    assert result.t_grid[1] - result.t_grid[0] == pytest.approx(dt)
    assert result.t_grid[-1] == pytest.approx(config.nt() * dt)
