"""
pee.py
====================================
Parameter error estimator functionality.
"""

import numpy as np
from scipy.optimize import minimize
from numdifftools import Jacobian


def parameter_error_estimator(
    fun,
    x_data,
    y_data,
    x_err,
    y_err,
    w_0,
    iter_num=None,
    rtol=None,
    atol=None,
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
):
    """[summary]

    Args:
        fun ([type]): [description]
        x_data ([type]): [description]
        y_data ([type]): [description]
        x_err ([type]): [description]
        y_err ([type]): [description]
        w_0 ([type]): [description]
        iter_num (int, optional): [description]. Defaults to 1000.
        method ([type], optional): [description]. Defaults to None.
        jac ([type], optional): [description]. Defaults to None.
        hess ([type], optional): [description]. Defaults to None.
        hessp ([type], optional): [description]. Defaults to None.
        bounds ([type], optional): [description]. Defaults to None.
        constraints (tuple, optional): [description]. Defaults to ().
        tol ([type], optional): [description]. Defaults to None.
        callback ([type], optional): [description]. Defaults to None.
        options ([type], optional): [description]. Defaults to None.
    """
    if iter_num is None and rtol is None and atol is None:
        raise TypeError("Argument iter_num or arguments rtol, atol have to be set.")

    if iter_num is None and (rtol is None or atol is None):
        raise TypeError("Both arguments rtol, atol have to be set.")

    def cost_fun(params, x, y):
        y_gen = np.array([fun(x_i, params) for x_i in x])
        return np.linalg.norm(y_gen - y)

    if method == "Newton-CG":

        def cost_fun_jac(params, x, y):
            return Jacobian(lambda p: cost_fun(p, x, y))(params).ravel()

    else:
        cost_fun_jac = None

    # print(cost_fun(w_0, x_data, y_data))

    result = minimize(
        cost_fun,
        w_0,
        args=(x_data, y_data),
        method=method,
        jac=cost_fun_jac,
        hess=hess,
        hessp=hessp,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        callback=callback,
        options=options,
    )

    if iter_num is not None:
        params_agg = np.zeros((iter_num, result.x.shape[0]), dtype=result.x.dtype)
        n, p_mean, M2 = 0, np.zeros_like(w_0), np.zeros_like(w_0)

        while n < iter_num:
            x_data_loc = x_data + np.random.normal(loc=0.0, scale=x_err)
            y_data_loc = y_data + np.random.normal(loc=0.0, scale=y_err)

            result = minimize(
                cost_fun,
                w_0,
                args=(x_data_loc, y_data_loc),
                method=method,
                jac=cost_fun_jac,
                hess=hess,
                hessp=hessp,
                bounds=bounds,
                constraints=constraints,
                tol=tol,
                callback=callback,
                options=options,
            )
            if result.success:
                # params_agg[n] = result.x
                n += 1
                delta = result.x - p_mean
                p_mean = p_mean + delta / n
                M2 = M2 + np.multiply(delta, result.x - p_mean)

    else:
        params_agg = []
        p_mean, p_std = parameter_error_estimator(
            fun,
            x_data,
            y_data,
            x_err,
            y_err,
            w_0,
            iter_num=3,
            rtol=None,
            atol=None,
            method=method,
            jac=cost_fun_jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options,
        )
        p_mean_prev, p_std_prev = 2 * p_mean, 2 * p_std
        variance = np.zeros_like(p_std)
        n, M2 = 3, np.zeros_like(w_0)

        while not (
            np.allclose(p_mean, p_mean_prev, rtol=rtol, atol=atol)
            and np.allclose(p_std, p_std_prev, rtol=rtol, atol=atol)
        ):
            x_data_loc = x_data + np.random.normal(loc=0.0, scale=x_err)
            y_data_loc = y_data + np.random.normal(loc=0.0, scale=y_err)

            result = minimize(
                cost_fun,
                w_0,
                args=(x_data_loc, y_data_loc),
                method=method,
                jac=cost_fun_jac,
                hess=hess,
                hessp=hessp,
                bounds=bounds,
                constraints=constraints,
                tol=tol,
                callback=callback,
                options=options,
            )
            if result.success:
                # params_agg.append(result.x)
                n += 1
                p_mean_prev, p_std_prev = p_mean, p_std
                delta = result.x - p_mean
                p_mean = p_mean + delta / n
                M2 = M2 + np.multiply(delta, result.x - p_mean)
                variance = M2 / (n - 1)
                p_std = np.sqrt(variance)

    variance = M2 / (n - 1)

    return p_mean, np.sqrt(variance)
