from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ._analytical import deming_analytical_solve
from ._mc import mc_solve
from .base import BaseRegressor


class DemingRegressor(BaseRegressor):
    """Regression estimator using Deming (total least squares) joint optimisation.

    Optimises jointly over model parameters and latent true x values, giving an
    exact treatment of both x and y measurement errors. Slower than
    ``XYWeightedRegressor`` but more accurate when x errors are large or the
    model is strongly nonlinear.

    Supports two solvers selected via the ``method`` argument:

    - ``"analytical"`` — joint optimisation with ``(J^T W J)^{-1}`` covariance.
    - ``"mc"`` — Monte Carlo sampling with Welford online covariance
      (default, robust for nonlinear models).

    Parameters:
        func: Model function with signature ``func(x, params) -> float``, where
            ``x`` is a scalar for 1-D input or a length-k array for k-dimensional
            input.
        method: Solver to use, either ``"analytical"`` or ``"mc"``. Default ``"mc"``.
        n_iter: Maximum number of Monte Carlo iterations. Default ``10_000``.
        rtol: Relative tolerance for MC convergence stopping. Default ``None`` (disabled).
        atol: Absolute tolerance for MC convergence stopping. Default ``None`` (disabled).
        optimizer: SciPy optimizer name used for parameter fitting. Default ``"BFGS"``.

    Attributes:
        params_: Fitted parameter array.
        params_std_: Standard deviations of fitted parameters.
        covariance_: Full parameter covariance matrix.
        n_iter_: Actual number of MC iterations run (MC method only).

    Notes:
        ``X`` may be shape ``(n,)`` for scalar input or ``(n, k)`` for k-dimensional
        input per data point.  ``x_err`` must match ``X.shape`` exactly; set a column
        to zero for any feature that is known exactly.  Zero-error features are pinned
        to their observed values inside the joint optimisation — they are excluded from
        the latent-variable cost to avoid division by zero.
    """

    def __init__(
        self,
        func: Callable,
        method: str = "mc",
        n_iter: int = 10_000,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        optimizer: str = "BFGS",
    ) -> None:
        super().__init__(
            func,
            method=method,
            n_iter=n_iter,
            rtol=rtol,
            atol=atol,
            optimizer=optimizer,
        )

    def fit(  # type: ignore[override]
        self,
        X: np.ndarray,
        y: np.ndarray,
        x_err: np.ndarray,
        y_err: np.ndarray,
        p0: np.ndarray,
    ) -> "DemingRegressor":
        X, y, y_err, x_err = self._validate_inputs(X, y, y_err, x_err)  # type: ignore[misc]
        p0 = np.asarray(p0, dtype=float)
        n_beta = len(p0)
        n_data = len(y)

        if self.method == "analytical":
            params, cov = deming_analytical_solve(
                self.func, X, y, x_err, y_err, p0, self.optimizer
            )
            self.params_ = params
            self.covariance_ = cov
            self.params_std_ = np.sqrt(np.diag(cov))
        else:
            x_var: np.ndarray = x_err**2  # type: ignore[assignment]
            y_var: np.ndarray = y_err**2  # type: ignore[assignment]
            # Zero-error features are known exactly: pin them to observed values
            # and exclude them from the x-residual cost to avoid division by zero.
            x_free: np.ndarray = x_var > 0  # type: ignore[assignment]
            x_var_safe: np.ndarray = np.where(x_free, x_var, 1.0)  # type: ignore[assignment]

            def cost_fn_builder(
                x_s: np.ndarray, y_s: np.ndarray, params_est: np.ndarray
            ) -> object:
                def cost(theta: np.ndarray) -> float:
                    beta = theta[:n_beta]
                    eta = theta[n_beta:].reshape(X.shape)
                    eta_eval = np.where(x_free, eta, x_s)
                    x_sq = np.where(x_free, (x_s - eta) ** 2 / x_var_safe, 0.0)
                    x_term: float = float(np.sum(x_sq))
                    y_hat = np.array([self.func(eta_eval[i], beta) for i in range(n_data)])
                    y_term: float = float(
                        np.sum(
                            (y_s - y_hat) ** 2
                            / y_var
                        )
                    )
                    return x_term + y_term

                return cost

            def extract_params(theta: np.ndarray) -> np.ndarray:
                return theta[:n_beta]

            def p0_fn(x_s: np.ndarray, y_s: np.ndarray) -> np.ndarray:
                return np.concatenate([p0, x_s.ravel()])  # type: ignore[no-any-return]

            theta0 = np.concatenate([p0, X.ravel()])

            mean, cov, n = mc_solve(
                cost_fn_builder,
                X,
                y,
                x_err,
                y_err,
                theta0,
                self.n_iter,
                self.rtol,
                self.atol,
                self.optimizer,
                extract_params=extract_params,
                p0_fn=p0_fn,
            )
            self.params_ = mean
            self.covariance_ = cov
            self.params_std_ = np.sqrt(np.diag(cov))
            self.n_iter_ = n

        return self
