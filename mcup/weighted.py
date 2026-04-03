from __future__ import annotations

import numpy as np

from ._analytical import analytical_solve
from ._mc import mc_solve
from .base import BaseRegressor


class WeightedRegressor(BaseRegressor):
    """Regression estimator for data where only y has measurement errors.

    Minimises the weighted chi-squared objective ``Σ (y - f(x))² / σ_y²``.
    Supports two solvers selected via the ``method`` argument:

    - ``"analytical"`` — weighted least squares with ``(J^T W J)^{-1}`` covariance (fast).
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
        y_err: np.ndarray,
        p0: np.ndarray,
    ) -> "WeightedRegressor":
        X, y, y_err = self._validate_inputs(X, y, y_err)  # type: ignore[misc]
        p0 = np.asarray(p0, dtype=float)
        self._y_err_fit_ = y_err

        weights = 1.0 / (y_err**2)

        if self.method == "analytical":
            params, cov = analytical_solve(self.func, X, y, weights, p0, self.optimizer)
            self.params_ = params
            self.covariance_ = cov
            self.params_std_ = np.sqrt(np.diag(cov))
        else:

            def cost_fn_builder(
                x_s: np.ndarray, y_s: np.ndarray, params_est: np.ndarray
            ) -> object:
                w = 1.0 / (y_err**2)

                def cost(params: np.ndarray) -> float:
                    r = np.array([y_s[i] - self.func(x_s[i], params) for i in range(len(y_s))])
                    return float(np.dot(r**2, w))

                return cost

            mean, cov, n = mc_solve(
                cost_fn_builder,
                X,
                y,
                None,
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
