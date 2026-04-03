from __future__ import annotations

from typing import Callable

import numpy as np
from numdifftools import Gradient

from ._analytical import analytical_solve
from ._mc import mc_solve
from .base import BaseRegressor


def _combined_weights(
    func: Callable,
    X: np.ndarray,
    params: np.ndarray,
    x_err: np.ndarray,
    y_err: np.ndarray,
) -> np.ndarray:
    var: np.ndarray = y_err**2  # type: ignore[assignment]
    for i in range(len(X)):
        xi = np.atleast_1d(X[i])
        df_dx = Gradient(lambda x: func(x, params))(xi)
        xe = np.atleast_1d(x_err[i])
        var[i] += float(np.dot(df_dx.ravel() ** 2, xe.ravel() ** 2))
    return 1.0 / var  # type: ignore[no-any-return]


class XYWeightedRegressor(BaseRegressor):
    """Regression estimator for data where both x and y have measurement errors.

    Uses iteratively reweighted least squares (IRLS) to combine x and y variances
    via error propagation: ``σ_combined² = σ_y² + (∂f/∂x)² σ_x²``. Faster than
    ``DemingRegressor`` and well-suited to mildly nonlinear models.

    Supports two solvers selected via the ``method`` argument:

    - ``"analytical"`` — IRLS with ``(J^T W J)^{-1}`` covariance (fast).
    - ``"mc"`` — Monte Carlo sampling with Welford online covariance (robust for nonlinear models).

    Parameters:
        func: Model function with signature ``func(x, params) -> float``.
        method: Solver to use, either ``"analytical"`` or ``"mc"``. Default ``"mc"``.
        n_iter: Maximum number of Monte Carlo iterations. Default ``10_000``.
        rtol: Relative tolerance for MC convergence stopping. Default ``None`` (disabled).
        atol: Absolute tolerance for MC convergence stopping. Default ``None`` (disabled).
        optimizer: SciPy optimizer name used for parameter fitting. Default ``"Nelder-Mead"``.

    Attributes:
        params_: Fitted parameter array.
        params_std_: Standard deviations of fitted parameters.
        covariance_: Full parameter covariance matrix.
        n_iter_: Actual number of MC iterations run (MC method only).
    """

    def fit(  # type: ignore[override]
        self,
        X: np.ndarray,
        y: np.ndarray,
        x_err: np.ndarray,
        y_err: np.ndarray,
        p0: np.ndarray,
        n_irls: int = 10,
    ) -> "XYWeightedRegressor":
        X, y, y_err, x_err = self._validate_inputs(X, y, y_err, x_err)  # type: ignore[misc]
        p0 = np.asarray(p0, dtype=float)

        if self.method == "analytical":
            params = p0.copy()
            for _ in range(n_irls):
                weights = _combined_weights(self.func, X, params, x_err, y_err)
                params, cov = analytical_solve(self.func, X, y, weights, params, self.optimizer)
            self.params_ = params
            self.covariance_ = cov
            self.params_std_ = np.sqrt(np.diag(cov))
        else:

            def cost_fn_builder(
                x_s: np.ndarray, y_s: np.ndarray, params_est: np.ndarray
            ) -> object:
                weights = _combined_weights(self.func, x_s, params_est, x_err, y_err)

                def cost(params: np.ndarray) -> float:
                    r = np.array([y_s[i] - self.func(x_s[i], params) for i in range(len(y_s))])
                    return float(np.dot(r**2, weights))

                return cost

            mean, cov, n = mc_solve(
                cost_fn_builder,
                X,
                y,
                x_err,
                y_err,
                p0,
                self.n_iter,
                self.rtol,
                self.atol,
                self.optimizer,
            )
            self.params_ = mean
            self.covariance_ = cov
            self.params_std_ = np.sqrt(np.diag(cov))
            self.n_iter_ = n

        return self
