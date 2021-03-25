"""
config.py
====================================
Setting for the package and pregenerated data.
"""

import numpy as np


def linear_fun(x, a, b):
    return a * x + b


N = 10
interval = [0, 10]
fun_params = [1.0, 2.0]
y_err_params = [0.1]
x_data = np.linspace(interval[0], interval[1], num=N, endpoint=True)
y_data = linear_fun(x_data, fun_params[0], fun_params[1])
# y_data = add_noise(add_noise)
