import numpy as np
from scipy.optimize import minimize

from ._utils import welford_update, welford_finalize


def mc_solve(
    cost_fn_builder,
    X,
    y,
    x_err,
    y_err,
    p0,
    n_iter,
    rtol,
    atol,
    optimizer,
    extract_params=None,
    p0_fn=None,
):
    if extract_params is None:
        extract_params = lambda theta: theta
    if p0_fn is None:
        p0_fn = lambda x_s, y_s: p0

    n_tracked = len(extract_params(p0))
    n, mean, cov_agg = 0, np.zeros(n_tracked), np.zeros((n_tracked, n_tracked))
    current_est = extract_params(p0).copy()

    def _step(x_s, y_s):
        nonlocal n, mean, cov_agg, current_est
        cost = cost_fn_builder(x_s, y_s, current_est)
        result = minimize(cost, p0_fn(x_s, y_s), method=optimizer)
        if result.success:
            tracked = extract_params(result.x)
            n, mean, cov_agg = welford_update(n, mean, cov_agg, tracked)
            current_est = mean.copy()

    x_noise_shape = X.shape
    y_noise_shape = y.shape

    if rtol is not None and atol is not None:
        mean_prev = np.full(n_tracked, np.inf)
        std_prev = np.full(n_tracked, np.inf)
        max_iter = n_iter if n_iter is not None else 100_000
        for _ in range(max_iter):
            x_s = X + np.random.normal(0, 1, x_noise_shape) * x_err if x_err is not None else X.copy()
            y_s = y + np.random.normal(0, 1, y_noise_shape) * y_err
            _step(x_s, y_s)
            if n > 1:
                std = np.sqrt(np.diag(welford_finalize(n, cov_agg)))
                if (
                    np.allclose(mean, mean_prev, rtol=rtol, atol=atol)
                    and np.allclose(std, std_prev, rtol=rtol, atol=atol)
                ):
                    break
                mean_prev, std_prev = mean.copy(), std.copy()
    else:
        for _ in range(n_iter):
            x_s = X + np.random.normal(0, 1, x_noise_shape) * x_err if x_err is not None else X.copy()
            y_s = y + np.random.normal(0, 1, y_noise_shape) * y_err
            _step(x_s, y_s)

    return mean, welford_finalize(n, cov_agg), n
