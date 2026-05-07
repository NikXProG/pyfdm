from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViscosityConfig:
    x_min: float = 0.0
    x_max: float = 25.0
    t_final: float = 6.0
    dx: float = 0.05
    a: float = 1.0
    mu: float = 0.8

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.dx <= 0.0:
            raise ValueError("dx must be positive.")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive.")
        if self.a == 0.0:
            raise ValueError("a must be non-zero.")
        if not (0.0 < self.mu <= 1.0):
            raise ValueError("mu must be in (0, 1].")

    @property
    def dt(self) -> float:
        return self.mu * self.dx / self.a

    @property
    def nt(self) -> int:
        return int(round(self.t_final / self.dt))

    @property
    def epsilon_num(self) -> float:
        return 0.5 * self.a * self.dx * (1.0 - self.mu)

    def x_grid(self) -> np.ndarray:
        nx = int(round((self.x_max - self.x_min) / self.dx)) + 1
        return np.linspace(self.x_min, self.x_max, nx)
