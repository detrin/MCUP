"""
config.py
====================================
Setting for the package and pregenerated data.
"""

import numpy as np
from .data_generator import DataGenerator
from .pee import parameter_error_estimator


def linear_fun(x, c):
    return c[0] * x + c[1]


data_len = 10
boundaries = [1.0, 10.0]
params = [1, 0]
params_0 = [0, 0]
y_err = 0.1 * np.ones(data_len)
x_err = np.zeros_like(y_err)
datagen = DataGenerator(linear_fun, data_len, boundaries, params=params)
x_data = datagen.x
y_data = datagen.add_noise_y(const_err=y_err)

# print(x_data)
# print(y_data)

params, err = parameter_error_estimator(
    linear_fun,
    x_data,
    y_data,
    x_err,
    y_err,
    params_0,
    iter_num=100,
    method="Nelder-Mead",
)
