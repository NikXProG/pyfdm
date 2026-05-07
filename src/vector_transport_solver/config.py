from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VectorTransportConfig:
    x_min: float = 0.0
    x_max: float = 6.0
    dx: float = 0.03
    t_final: float = 20.0
    cfl: float = 0.85
    animation_stride: int = 10

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.dx <= 0.0:
            raise ValueError("dx must be positive.")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive.")
        if self.cfl <= 0.0:
            raise ValueError("cfl must be positive.")
        if self.animation_stride < 1:
            raise ValueError("animation_stride must be >= 1.")

    @property
    def matrix_a(self) -> np.ndarray:
        return np.array([[-1.0, 2.0], [1.5, 1.0]], dtype=float)

    def eigen_system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vals, vecs = np.linalg.eig(self.matrix_a)
        if np.any(np.abs(vals.imag) > 1e-12):
            raise ValueError("matrix_a has complex eigenvalues; real characteristic transform is invalid.")
        if np.any(np.abs(vecs.imag) > 1e-12):
            raise ValueError("matrix_a eigenvectors are complex; unsupported for this solver.")
        inv_vecs = np.linalg.inv(vecs)
        return vals.real, vecs.real, inv_vecs.real

    def x_grid(self) -> np.ndarray:
        nx = int(round((self.x_max - self.x_min) / self.dx)) + 1
        return np.linspace(self.x_min, self.x_max, nx)

    def dt(self) -> float:
        lambdas, _, _ = self.eigen_system()
        max_speed = float(np.max(np.abs(lambdas)))
        return self.cfl * self.dx / max_speed

    def nt(self) -> int:
        return int(np.ceil(self.t_final / self.dt()))
