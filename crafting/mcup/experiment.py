"""Generic data generator."""

import numpy as np

class VirtualExperiment(object):
    def __init__(
            self, fun, x_bounds=None, sample_num=None, x_error=0, y_error=0, params=(),
            kwargs={}):
        '''Seting basic properties of virtual experiment.'''
        if not callable(fun):
            raise TypeError("`fun` must be callable.")

        if len(x_bounds) != 2:
            raise ValueError("`x_bounds` must contain 2 elements.")
        
        if not isinstance(sample_num, int):
            raise TypeError("`sample_num` must be integer.")

        if not (isinstance(x_error, float) or isinstance(x_error, int)):
            raise TypeError("`x_error` must be integer or float.")

        if x_error < 0:
            raise ValueError("`x_error` must be non-negative.")

        if not (isinstance(y_error, float) or isinstance(y_error, int)):
            raise TypeError("`y_error` must be integer or float.")

        if y_error < 0:
            raise ValueError("`y_error` must be non-negative.")

        if np.iscomplexobj(params):
            raise ValueError("`params` must be real.")

        params = np.atleast_1d(params).astype(float)

        if params.ndim > 1:
            raise ValueError("`params` must have at most 1 dimension.")
        
        self.x_error = x_error
        self.y_error = y_error
        self.x = np.linspace(x_bounds[0], x_bounds[1], sample_num)
        self.params = params
        self.kwargs = kwargs
        self.fun = fun
        self.sample_num = sample_num

        f0 = self.fun_wrapped(self.x)
        if f0.ndim != 1:
            raise ValueError("`fun` must return at most 1-d array_like. "
                            "f0.shape: {0}".format(f0.shape))

    def fun_wrapped(self, x):
        return np.atleast_1d(self.fun(x, *self.params, **self.kwargs))

    def measure(self):
        '''Measure experiment with given uncertainty.'''
        x = self.x + np.random.normal(0, self.x_error)
        y = self.fun_wrapped(self.x) + np.random.normal(0, self.y_error)
        x_errs = self.x_error*np.ones(self.sample_num)
        y_errs = self.y_error*np.ones(self.sample_num)

        return [x, y, x_errs, y_errs]

        