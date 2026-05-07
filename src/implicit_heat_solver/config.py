from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImplicitHeatConfig:
    x_min: float = -5.0
    x_max: float = 5.0
    dx: float = 0.05
    t_final: float = 3.0
    dt: float = 0.01
    kappa: float = 0.25  # 4u_t = u_xx -> u_t = (1/4)u_xx

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.dx <= 0.0 or self.dt <= 0.0:
            raise ValueError("dx and dt must be positive.")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive.")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive.")

    def x_grid(self) -> np.ndarray:
        nx = int(round((self.x_max - self.x_min) / self.dx)) + 1
        return np.linspace(self.x_min, self.x_max, nx)

    def t_grid(self) -> np.ndarray:
        nt = int(round(self.t_final / self.dt))
        return np.linspace(0.0, self.t_final, nt + 1)

    def snapshot_times(self) -> np.ndarray:
        return np.arange(0.0, self.t_final + 0.25, 0.5)
