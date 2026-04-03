from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

from ._utils import welford_finalize, welford_update


def mc_solve(
    cost_fn_builder: Callable,
    X: np.ndarray,
    y: np.ndarray,
    x_err: Optional[np.ndarray],
    y_err: np.ndarray,
    p0: np.ndarray,
    n_iter: int,
    rtol: Optional[float],
    atol: Optional[float],
    optimizer: str,
    extract_params: Optional[Callable] = None,
    p0_fn: Optional[Callable] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    def _default_extract(theta: np.ndarray) -> np.ndarray:
        return theta

    def _default_p0_fn(x_s: np.ndarray, y_s: np.ndarray) -> np.ndarray:
        return p0

    _extract: Callable = extract_params if extract_params is not None else _default_extract
    _p0_fn: Callable = p0_fn if p0_fn is not None else _default_p0_fn

    n_tracked = len(_extract(p0))
    n, mean, cov_agg = 0, np.zeros(n_tracked), np.zeros((n_tracked, n_tracked))
    current_est = _extract(p0).copy()

    def _step(x_s: np.ndarray, y_s: np.ndarray) -> None:
        nonlocal n, mean, cov_agg, current_est
        cost = cost_fn_builder(x_s, y_s, current_est)
        result = minimize(cost, _p0_fn(x_s, y_s), method=optimizer)
        if result.success:
            tracked = _extract(result.x)
            n, mean, cov_agg = welford_update(n, mean, cov_agg, tracked)
            current_est = mean.copy()

    x_noise_shape = X.shape
    y_noise_shape = y.shape

    if rtol is not None and atol is not None:
        mean_prev = np.full(n_tracked, np.inf)
        std_prev = np.full(n_tracked, np.inf)
        max_iter = n_iter if n_iter is not None else 100_000
        for _ in range(max_iter):
            if x_err is not None:
                x_s = X + np.random.normal(0, 1, x_noise_shape) * x_err
            else:
                x_s = X.copy()
            y_s = y + np.random.normal(0, 1, y_noise_shape) * y_err
            _step(x_s, y_s)
            if n > 1:
                std = np.sqrt(np.diag(welford_finalize(n, cov_agg)))
                if np.allclose(mean, mean_prev, rtol=rtol, atol=atol) and np.allclose(
                    std, std_prev, rtol=rtol, atol=atol
                ):
                    break
                mean_prev, std_prev = mean.copy(), std.copy()
    else:
        for _ in range(n_iter):
            if x_err is not None:
                x_s = X + np.random.normal(0, 1, x_noise_shape) * x_err
            else:
                x_s = X.copy()
            y_s = y + np.random.normal(0, 1, y_noise_shape) * y_err
            _step(x_s, y_s)

    return mean, welford_finalize(n, cov_agg), n
