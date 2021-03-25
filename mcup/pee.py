"""
pee.py
====================================
Parameter error estimator functionality.
"""

import numpy as np
from scipy.optimize import minimize


def parameter_error_estimator(
    fun,
    x_data,
    y_data,
    x_err,
    y_err,
    w_0,
    iter_num=1000,
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

    def cost_fun(params, x, y):
        return np.linalg.norm(fun(x, params) - y)

    result = minimize(
        cost_fun,
        w_0,
        args=(x_data, y_data),
        method=method,
        jac=jac,
        hess=hess,
        hessp=hessp,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        callback=callback,
        options=options,
    )
    assert result.success

    params_agg = np.zeros((iter_num, result.x.shape[0]), dtype=result.x.dtype)
    run_i = 0
    while run_i < iter_num:
        x_data_loc = x_data + np.random.normal(loc=0.0, scale=x_err)
        y_data_loc = y_data + np.random.normal(loc=0.0, scale=y_err)

        result = minimize(
            cost_fun,
            w_0,
            args=(x_data_loc, y_data_loc),
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options,
        )
        if result.success:
            params_agg[run_i] = result.x
            run_i += 1

    return np.mean(params_agg, axis=0), np.std(params_agg, axis=0)
