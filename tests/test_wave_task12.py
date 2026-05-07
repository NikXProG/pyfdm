import pytest

from wave_task12_solver.config import WaveTask12Config
from wave_task12_solver.physics import initial_displacement, initial_velocity


def test_wave_task12_initial_data_matches_neumann_boundaries() -> None:
    config = WaveTask12Config()
    x = config.x_grid()
    u0 = initial_displacement(x)
    v0 = initial_velocity(x)
    ux_left = (u0[1] - u0[0]) / config.dx
    ux_right = (u0[-1] - u0[-2]) / config.dx
    vx_left = (v0[1] - v0[0]) / config.dx
    vx_right = (v0[-1] - v0[-2]) / config.dx
    assert ux_left == pytest.approx(0.0, abs=0.5)
    assert ux_right == pytest.approx(0.0, abs=0.5)
    assert vx_left == pytest.approx(0.0, abs=0.3)
    assert vx_right == pytest.approx(0.0, abs=0.3)
