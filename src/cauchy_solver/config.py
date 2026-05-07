from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    x_min: float = 0.0
    x_max: float = 1.0
    t_final: float = 5.0
    nx: int = 201
    nt: int = 800

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive.")
        if self.nx < 3:
            raise ValueError("nx must be at least 3.")
        if self.nt < 1:
            raise ValueError("nt must be at least 1.")

    def x_grid(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.nx)

    def t_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.t_final, self.nt + 1)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def dt(self) -> float:
        return self.t_final / self.nt
