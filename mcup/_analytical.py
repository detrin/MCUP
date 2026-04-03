import numpy as np
from scipy.optimize import minimize
from numdifftools import Jacobian


def analytical_solve(func, X, y, weights, p0, optimizer):
    W = np.diag(weights)

    def cost(params):
        r = np.array([y[i] - func(X[i], params) for i in range(len(y))])
        return float(r @ W @ r)

    result = minimize(cost, p0, method=optimizer)
    params = result.x

    J = Jacobian(lambda p: np.array([func(X[i], p) for i in range(len(X))]))(params)
    cov = np.linalg.inv(J.T @ W @ J)
    return params, cov


def deming_analytical_solve(func, X_obs, y_obs, x_err, y_err, p0, optimizer):
    n_beta = len(p0)
    n = len(y_obs)
    x_var = x_err ** 2
    y_var = y_err ** 2
    theta0 = np.concatenate([p0, X_obs.ravel()])

    def cost(theta):
        beta = theta[:n_beta]
        eta = theta[n_beta:].reshape(X_obs.shape)
        x_term = np.sum((X_obs - eta) ** 2 / x_var)
        y_term = np.sum(
            (y_obs - np.array([func(eta[i], beta) for i in range(n)])) ** 2 / y_var
        )
        return x_term + y_term

    def residuals(theta):
        beta = theta[:n_beta]
        eta = theta[n_beta:].reshape(X_obs.shape)
        r_x = (X_obs - eta) / x_err
        r_y = (y_obs - np.array([func(eta[i], beta) for i in range(n)])) / y_err
        return np.concatenate([r_x.ravel(), r_y.ravel()])

    result = minimize(cost, theta0, method=optimizer)
    theta = result.x
    beta = theta[:n_beta]

    J = Jacobian(residuals)(theta)  # shape (2n, n_beta + n)
    try:
        full_cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        full_cov = np.full((len(theta0), len(theta0)), np.nan)

    cov = full_cov[:n_beta, :n_beta]
    return beta, cov
