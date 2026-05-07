import numpy as np


def to_characteristic(u: np.ndarray, inv_vecs: np.ndarray) -> np.ndarray:
    return inv_vecs @ u


def from_characteristic(z: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    return vecs @ z
