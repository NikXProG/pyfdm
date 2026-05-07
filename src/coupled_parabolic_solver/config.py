from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoupledConfig:
    x_min: float = 0.0
    x_max: float = 1.0
    dx: float = 0.02
    t_final: float = 2.0
    dt: float = 0.002
    d0: float = 1.0
    d1: float = 1.0
    delta_e1: float = 1.0
    k: float = 1.0

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.dx <= 0.0 or self.dt <= 0.0:
            raise ValueError("dx and dt must be positive.")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive.")
        if self.d0 <= 0.0 or self.d1 <= 0.0:
            raise ValueError("d0 and d1 must be positive.")
        if self.k <= 0.0:
            raise ValueError("k must be positive.")
        if self.delta_e1 < 0.0:
            raise ValueError("delta_e1 must be non-negative.")

    def x_grid(self) -> np.ndarray:
        nx = int(round((self.x_max - self.x_min) / self.dx)) + 1
        return np.linspace(self.x_min, self.x_max, nx)

    def t_grid(self) -> np.ndarray:
        nt = int(round(self.t_final / self.dt))
        return np.linspace(0.0, self.t_final, nt + 1)

    @property
    def tol_fourier(self) -> float:
        return self.dx * self.dx

    def case_ids(self) -> tuple[str, ...]:
        return ("a", "b", "c")
