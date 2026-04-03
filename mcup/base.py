import numpy as np


class BaseRegressor:
    def __init__(self, func, method="mc", n_iter=10_000, rtol=None, atol=None, optimizer="Nelder-Mead"):
        self.func = func
        self.method = method
        self.n_iter = n_iter
        self.rtol = rtol
        self.atol = atol
        self.optimizer = optimizer

    def get_params(self, deep=True):
        return {
            "func": self.func,
            "method": self.method,
            "n_iter": self.n_iter,
            "rtol": self.rtol,
            "atol": self.atol,
            "optimizer": self.optimizer,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "params_"):
            raise ValueError("Estimator is not fitted. Call fit() first.")

    def _validate_inputs(self, X, y, y_err, x_err=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        y_err = np.asarray(y_err, dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if y_err.shape != y.shape:
            raise ValueError("y_err must have the same shape as y.")
        if x_err is not None:
            x_err = np.asarray(x_err, dtype=float)
            if x_err.shape != X.shape:
                raise ValueError("x_err must have the same shape as X.")
            return X, y, y_err, x_err
        return X, y, y_err

    def fit(self, X, y, **kwargs):
        raise NotImplementedError
