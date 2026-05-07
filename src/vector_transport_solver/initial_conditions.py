import numpy as np


def initial_v(x: np.ndarray) -> np.ndarray:
    # Интерпретация скана: трехуровневая кусочная структура для первой компоненты v.
    v = np.ones_like(x)
    v[(x > 2.0) & (x < 3.0)] = 0.2
    side = ((x > 1.0) & (x < 2.0)) | ((x > 3.0) & (x < 4.0))
    v[side] = 0.5
    return v


def initial_w(x: np.ndarray) -> np.ndarray:
    # Для второй компоненты w используем зеркальную трехуровневую структуру.
    w = np.full_like(x, 0.2)
    side = ((x > 1.0) & (x < 2.0)) | ((x > 3.0) & (x < 4.0))
    w[side] = 0.5
    w[(x > 2.0) & (x < 3.0)] = 1.0
    return w


def build_initial_u(x: np.ndarray) -> np.ndarray:
    return np.vstack([initial_v(x), initial_w(x)])
