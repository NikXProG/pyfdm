from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdvectionConfig:
    x_min: float = 0.0
    x_max: float = 25.0
    dx: float = 0.05
    mu: float = 0.8
    a: float = 1.0
    t_target: float = 17.0

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.dx <= 0.0:
            raise ValueError("dx must be positive.")
        if self.t_target <= 0.0:
            raise ValueError("t_target must be positive.")
        if self.a == 0.0:
            raise ValueError("a must be non-zero.")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive.")

    @property
    def dt(self) -> float:
        return self.mu * self.dx / abs(self.a)

    @property
    def nt(self) -> int:
        return int(round(self.t_target / self.dt))

    def x_grid(self) -> np.ndarray:
        nx = int(round((self.x_max - self.x_min) / self.dx)) + 1
        return np.linspace(self.x_min, self.x_max, nx)
