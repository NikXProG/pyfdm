from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Dataset:
    f: Callable[[np.ndarray], np.ndarray]
    u1: Callable[[np.ndarray], np.ndarray]
    u2: Callable[[np.ndarray], np.ndarray]
    t0: Callable[[np.ndarray], np.ndarray]
    u1_dt: Callable[[np.ndarray], np.ndarray]
    u2_dt: Callable[[np.ndarray], np.ndarray]


def _case_a() -> Dataset:
    return Dataset(
        f=lambda x: np.sin(np.pi * x),
        u1=lambda t: 2.0 * t,
        u2=lambda t: t * t,
        # Match boundary values at t=0 to avoid corner singularity in implicit solve.
        t0=lambda x: 4.0 * x * (1.0 - x) * np.exp(-0.5 * np.abs(x - 0.5)),
        u1_dt=lambda t: np.full_like(t, 2.0, dtype=float),
        u2_dt=lambda t: 2.0 * t,
    )


def _case_b() -> Dataset:
    return Dataset(
        f=lambda x: np.exp(-((2.0 * x - 1.0) ** 2) / 8.0),
        u1=lambda t: np.exp(-4.0 * t),
        u2=lambda t: np.exp(-5.0 * t),
        t0=lambda x: np.cos(np.pi * x) ** 2,
        u1_dt=lambda t: -4.0 * np.exp(-4.0 * t),
        u2_dt=lambda t: -5.0 * np.exp(-5.0 * t),
    )


def _case_c() -> Dataset:
    return Dataset(
        f=lambda x: np.where((x >= 0.0) & (x <= 0.2), 1.0, 0.0),
        u1=lambda t: np.exp(2.0 * t),
        u2=lambda t: -(t * t),
        t0=lambda x: np.where((x >= 0.0) & (x <= 0.3), 1.0, 0.0),
        u1_dt=lambda t: 2.0 * np.exp(2.0 * t),
        u2_dt=lambda t: -2.0 * t,
    )


def get_dataset(case_id: str) -> Dataset:
    cases = {"a": _case_a(), "b": _case_b(), "c": _case_c()}
    return cases[case_id]
