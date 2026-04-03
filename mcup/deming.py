import numpy as np

from .base import BaseRegressor
from ._analytical import deming_analytical_solve
from ._mc import mc_solve


class DemingRegressor(BaseRegressor):
    def __init__(self, func, method="mc", n_iter=10_000, rtol=None, atol=None, optimizer="BFGS"):
        super().__init__(func, method=method, n_iter=n_iter, rtol=rtol, atol=atol, optimizer=optimizer)

    def fit(self, X, y, x_err, y_err, p0):
        X, y, y_err, x_err = self._validate_inputs(X, y, y_err, x_err)
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
            x_var = x_err ** 2
            y_var = y_err ** 2

            def cost_fn_builder(x_s, y_s, params_est):
                def cost(theta):
                    beta = theta[:n_beta]
                    eta = theta[n_beta:].reshape(X.shape)
                    x_term = np.sum((x_s - eta) ** 2 / x_var)
                    y_term = np.sum(
                        (y_s - np.array([self.func(eta[i], beta) for i in range(n_data)])) ** 2 / y_var
                    )
                    return x_term + y_term
                return cost

            def extract_params(theta):
                return theta[:n_beta]

            def p0_fn(x_s, y_s):
                return np.concatenate([p0, x_s.ravel()])

            theta0 = np.concatenate([p0, X.ravel()])

            mean, cov, n = mc_solve(
                cost_fn_builder, X, y, x_err, y_err, theta0,
                self.n_iter, self.rtol, self.atol, self.optimizer,
                extract_params=extract_params,
                p0_fn=p0_fn,
            )
            self.params_ = mean
            self.covariance_ = cov
            self.params_std_ = np.sqrt(np.diag(cov))
            self.n_iter_ = n

        return self
