from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BurgersConfig:
    x_min: float = 0.0
    x_max: float = 15.0
    dx: float = 0.05
    t_final: float = 10.0
    cfl_conv: float = 0.4
    cfl_diff: float = 0.45
    eps_base: float = 0.5

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.dx <= 0.0:
            raise ValueError("dx must be positive.")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive.")
        if self.cfl_conv <= 0.0 or self.cfl_diff <= 0.0:
            raise ValueError("CFL coefficients must be positive.")
        if self.eps_base <= 0.0:
            raise ValueError("eps_base must be positive.")

    def x_grid(self) -> np.ndarray:
        nx = int(round((self.x_max - self.x_min) / self.dx)) + 1
        return np.linspace(self.x_min, self.x_max, nx)

    def epsilon_values(self) -> np.ndarray:
        return np.arange(0.1, 2.0 + 0.05, 0.1)
