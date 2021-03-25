"""
mcup.py
====================================
The core module of MCUP package.
"""

import copy
import numpy as np
from .pee import parameter_error_estimator


class Measurement:
    """An example docstring for a class definition."""

    def __init__(self, x=None, y=None, x_err=None, y_err=None):
        """[summary]

        Args:
            x ([type], optional): [description]. Defaults to None.
            y ([type], optional): [description]. Defaults to None.
            x_err ([type], optional): [description]. Defaults to None.
            y_err ([type], optional): [description]. Defaults to None.
        """
        if x is not None:
            self.set_data(x=x, y=y, x_err=x_err, y_err=y_err)

        self.params = None
        self.fun = None
        self.jac = None

    def set_data(self, x=None, y=None, x_err=None, y_err=None):
        """[summary]

        Args:
            x ([type], optional): [description]. Defaults to None.
            y ([type], optional): [description]. Defaults to None.
            x_err ([type], optional): [description]. Defaults to None.
            y_err ([type], optional): [description]. Defaults to None.

        Raises:
            AssertionError: [description]
            AssertionError: [description]
        """
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.x_err = copy.deepcopy(x_err)
        self.y_err = copy.deepcopy(y_err)

        if x is None or y is None:
            raise TypeError("To set Measurement data x, y have to be defined.")
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)

        if x_err is None:
            self.x_err = np.zeros_like(self.x)

        if y_err is None:
            self.y_err = np.zeros_like(self.y)

        def check_item(item, skip_length_check=False):
            if not isinstance(item, (list, np.ndarray)):
                raise TypeError("All argument have to be list or np.ndarray.")
            if isinstance(item, list):
                item = np.array(item)

            if not skip_length_check:
                if item.shape[0] != self.x.shape[0]:
                    raise TypeError("All argument have to have same length.")

            return item

        self.x = check_item(self.x, skip_length_check=True)
        self.y = check_item(self.y)
        self.x_err = check_item(self.x_err)
        self.y_err = check_item(self.y_err)

        self.data_len = self.x.shape[0]

        if self.x.ndim != self.x_err.ndim:
            raise TypeError("Arguments x and x_err have to have the same length.")

    def set_function(self, fun, params):
        if not callable(fun):
            raise TypeError("Argument fun has to be callable.")

        if not isinstance(params, (list, np.ndarray)):
            raise TypeError("Argument params has to be list or np.ndarray.")

        if isinstance(params, list):
            params = np.array(params)

        self.params = params
        self.fun = fun

        return True

    def evaluate_params(self, iter_num=None, rtol=1e-4, atol=1e-4, num_diff=False):
        if self.fun is None or self.params is None:
            raise RuntimeError("Function or params not set, use set_function().")

        self.fun(self.x[0], self.params)

        if not num_diff:
            params_mean, params_std = parameter_error_estimator(
                self.fun,
                self.x,
                self.y,
                self.x_err,
                self.y_err,
                self.params,
                iter_num=iter_num,
                rtol=rtol,
                atol=atol,
                method="Nelder-Mead",
            )
        else:
            params_mean, params_std = parameter_error_estimator(
                self.fun,
                self.x,
                self.y,
                self.x_err,
                self.y_err,
                self.params,
                iter_num=iter_num,
                rtol=rtol,
                atol=atol,
                method="Newton-CG",
            )

        return params_mean, params_std
