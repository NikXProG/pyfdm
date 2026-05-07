from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LWConfig:
    x_min: float = 0.0
    x_max: float = 25.0
    dx: float = 0.05
    a: float = 1.0
    t_final: float = 6.0
    mu_values: tuple[float, ...] = (0.8, 1.0, 1.2)

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.dx <= 0.0:
            raise ValueError("dx must be positive.")
        if self.a == 0.0:
            raise ValueError("a must be non-zero.")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive.")
        if not self.mu_values:
            raise ValueError("mu_values cannot be empty.")

    def x_grid(self) -> np.ndarray:
        nx = int(round((self.x_max - self.x_min) / self.dx)) + 1
        return np.linspace(self.x_min, self.x_max, nx)

    def dt(self, mu: float) -> float:
        return mu * self.dx / abs(self.a)

    def nt(self, mu: float) -> int:
        return int(round(self.t_final / self.dt(mu)))
