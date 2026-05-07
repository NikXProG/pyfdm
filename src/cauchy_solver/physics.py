import numpy as np


def diffusion_like_coeff(x: np.ndarray) -> np.ndarray:
    return np.cos(x) + 2.0


def reaction_coeff(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def initial_displacement(x: np.ndarray) -> np.ndarray:
    return 32.0 * x**2 * (1.0 - x) ** 3


def initial_velocity(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)
