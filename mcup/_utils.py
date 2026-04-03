import contextlib
import numpy as np


@contextlib.contextmanager
def local_numpy_seed(seed):
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            np.random.set_state(state)


def welford_update(n, mean, cov_agg, x):
    n = n + 1
    delta = x - mean
    mean = mean + delta / n
    cov_agg = cov_agg + np.outer(delta, x - mean)
    return n, mean, cov_agg


def welford_finalize(n, cov_agg):
    if n < 2:
        return np.full_like(cov_agg, np.nan)
    return cov_agg / (n - 1)
