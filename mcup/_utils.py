from __future__ import annotations

import contextlib
from typing import Generator, Optional

import numpy as np


@contextlib.contextmanager
def local_numpy_seed(seed: Optional[int]) -> Generator[None, None, None]:
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            np.random.set_state(state)


def welford_update(
    n: int,
    mean: np.ndarray,
    cov_agg: np.ndarray,
    x: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    n = n + 1
    delta = x - mean
    mean = mean + delta / n
    cov_agg = cov_agg + np.outer(delta, x - mean)
    return n, mean, cov_agg


def welford_finalize(n: int, cov_agg: np.ndarray) -> np.ndarray:
    if n < 2:
        return np.full_like(cov_agg, np.nan)  # type: ignore[no-any-return]
    return cov_agg / (n - 1)  # type: ignore[no-any-return]
