from __future__ import annotations

from typing import Callable

import numpy as np
from numdifftools import Jacobian
from scipy.optimize import minimize


def ols_solve(
    func: Callable,
    X: np.ndarray,
    y: np.ndarray,
    p0: np.ndarray,
    optimizer: str,
) -> tuple[np.ndarray, np.ndarray]:
    def cost(params: np.ndarray) -> float:
        r = np.array([y[i] - func(X[i], params) for i in range(len(y))])
        return float(r @ r)

    result = minimize(cost, p0, method=optimizer)
    params = result.x
    n, p = len(y), len(params)

    J = Jacobian(lambda q: np.array([func(X[i], q) for i in range(len(X))]))(params)
    residuals = np.array([y[i] - func(X[i], params) for i in range(len(y))])
    sigma2 = float(np.sum(residuals**2)) / max(n - p, 1)

    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((len(params), len(params)), np.nan)

    return params, cov


def analytical_solve(
    func: Callable,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    p0: np.ndarray,
    optimizer: str,
) -> tuple[np.ndarray, np.ndarray]:
    W = np.diag(weights)

    def cost(params: np.ndarray) -> float:
        r = np.array([y[i] - func(X[i], params) for i in range(len(y))])
        return float(r @ W @ r)

    result = minimize(cost, p0, method=optimizer)
    params = result.x

    J = Jacobian(lambda p: np.array([func(X[i], p) for i in range(len(X))]))(params)
    cov = np.linalg.inv(J.T @ W @ J)
    return params, cov


def deming_analytical_solve(
    func: Callable,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    x_err: np.ndarray,
    y_err: np.ndarray,
    p0: np.ndarray,
    optimizer: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_beta = len(p0)
    n = len(y_obs)
    x_var: np.ndarray = x_err**2  # type: ignore[assignment]
    y_var: np.ndarray = y_err**2  # type: ignore[assignment]
    # Mask for features that carry measurement error. Zero-error features are
    # treated as exactly known: they contribute zero to the cost and are pinned
    # to their observed values when evaluating the model.
    x_free: np.ndarray = x_var > 0  # type: ignore[assignment]
    x_var_safe: np.ndarray = np.where(x_free, x_var, 1.0)  # type: ignore[assignment]
    x_err_safe: np.ndarray = np.where(x_free, x_err, 1.0)  # type: ignore[assignment]
    theta0 = np.concatenate([p0, X_obs.ravel()])

    def cost(theta: np.ndarray) -> float:
        beta = theta[:n_beta]
        eta = theta[n_beta:].reshape(X_obs.shape)
        eta_eval = np.where(x_free, eta, X_obs)
        x_term: float = float(np.sum(np.where(x_free, (X_obs - eta) ** 2 / x_var_safe, 0.0)))
        y_term: float = float(
            np.sum((y_obs - np.array([func(eta_eval[i], beta) for i in range(n)])) ** 2 / y_var)
        )
        return x_term + y_term

    def residuals(theta: np.ndarray) -> np.ndarray:
        beta = theta[:n_beta]
        eta = theta[n_beta:].reshape(X_obs.shape)
        eta_eval = np.where(x_free, eta, X_obs)
        r_x = np.where(x_free, (X_obs - eta) / x_err_safe, 0.0)
        r_y = (y_obs - np.array([func(eta_eval[i], beta) for i in range(n)])) / y_err
        return np.concatenate([r_x.ravel(), r_y.ravel()])  # type: ignore[no-any-return]

    result = minimize(cost, theta0, method=optimizer)
    theta = result.x
    beta = theta[:n_beta]

    J = Jacobian(residuals)(theta)
    JtJ = J.T @ J
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        try:
            full_cov = np.linalg.inv(JtJ)
        except np.linalg.LinAlgError:
            full_cov = np.full((len(theta0), len(theta0)), np.nan)
    if not np.all(np.isfinite(full_cov)):
        # Zero-error features produce zero rows in J making J^T J singular.
        # The Moore-Penrose pseudoinverse gives a valid covariance for the
        # free-parameter subspace.
        full_cov = np.linalg.pinv(JtJ)

    cov = full_cov[:n_beta, :n_beta]
    return beta, cov
