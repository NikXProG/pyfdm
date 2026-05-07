from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SigmaHeatConfig:
    x_min: float = -6.0
    x_max: float = 6.0
    dx: float = 0.05
    t_final: float = 5.0
    r: float = 0.4  # tau/h^2
    sigma_values: tuple[float, ...] = (0.0, 0.5, 1.0)

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.dx <= 0.0:
            raise ValueError("dx must be positive.")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive.")
        if self.r <= 0.0:
            raise ValueError("r must be positive.")
        if not self.sigma_values:
            raise ValueError("sigma_values cannot be empty.")

    @property
    def dt(self) -> float:
        return self.r * self.dx * self.dx

    def x_grid(self) -> np.ndarray:
        nx = int(round((self.x_max - self.x_min) / self.dx)) + 1
        return np.linspace(self.x_min, self.x_max, nx)

    def t_grid(self) -> np.ndarray:
        nt = int(round(self.t_final / self.dt))
        return np.linspace(0.0, self.t_final, nt + 1)
