from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecalculationHeatConfig:
    x_min: float = -6.0
    x_max: float = 6.0
    dx: float = 0.05
    t_final: float = 7.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.dx <= 0.0 or self.dt <= 0.0:
            raise ValueError("dx and dt must be positive.")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive.")

    def x_grid(self) -> np.ndarray:
        nx = int(round((self.x_max - self.x_min) / self.dx)) + 1
        return np.linspace(self.x_min, self.x_max, nx)

    def t_grid(self) -> np.ndarray:
        nt = int(round(self.t_final / self.dt))
        return np.linspace(0.0, self.t_final, nt + 1)
