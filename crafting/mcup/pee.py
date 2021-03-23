"""Propagartor Error Estimator file."""

import numpy as np
from multiprocessing import Pool
from contextlib import closing

from .lsq import LeastSquares


class PropagatorErrorEstimator(object):
    def __init__(self, x_data, y_data, x_err, y_err, w_0):
        # TODO: add exceptions later
        self.x = np.array(x_data)
        self.y = np.array(y_data)
        self.xe = np.array(x_err)
        self.ye = np.array(y_err)
        self.w_0 = w_0
        self.data_len = len(x_data)
        self.param_len = len(w_0)
        # decide how to select correct dtype from x, y, x_errs, y_errs
        self.dtype = self.x.dtype
        self.var_dist = None
        self.w_dist = None
        self.fun = None
        self.lsq = None
        self.lsq_method = "normal"

    def Core(self, run_i):
        x_difference = np.zeros((self.data_len), dtype=self.dtype)
        y_difference = np.zeros((self.data_len), dtype=self.dtype)
        for ind in range(self.data_len):
            x_difference[ind] = np.random.normal(0, self.xe[ind])
            y_difference[ind] = np.random.normal(0, self.ye[ind])
        x_new = self.x + x_difference
        y_new = self.y + y_difference
        w, var = self.lsq.fit(self.fun, x_new, y_new, self.w_0, self.lsq_method)
        return [w, var]

    def run(self, lsq, fun, sample_num=10 ** 4, n_thread=1, method="normal"):
        if not isinstance(lsq, LeastSquares):
            raise TypeError("`lsq` must be LSQ object.")

        self.lsq = lsq
        self.fun = fun
        self.var_dist = np.zeros((sample_num, self.param_len), dtype=self.dtype)
        self.w_dist = np.zeros((sample_num, self.param_len), dtype=self.dtype)
        self.lsq_method = method

        if n_thread == 1:
            for run_i in range(sample_num):
                self.w_dist[run_i], self.var_dist[run_i] = self.Core(run_i)
        else:
            with closing(Pool(processes=n_thread)) as pool:
                result = pool.map(self.Core, range(sample_num))
                for res_i in range(sample_num):
                    self.w_dist[res_i], self.var_dist[res_i] = result[res_i]

        return [self.w_dist, self.var_dist]
