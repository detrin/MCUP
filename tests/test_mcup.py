import unittest
import numpy as np

from mcup import Measurement
from mcup import DataGenerator


class TestMeasurement(unittest.TestCase):
    def test_set_data(self):
        def linear_fun(x, c):
            return c[0] * x + c[1]

        data_len = 10
        boundaries = [1.0, 10.0]
        params = [1, 0]
        x_err = 0.1 * np.ones(data_len)
        y_err = 0.1 * np.ones(data_len)
        datagen = DataGenerator(linear_fun, data_len, boundaries, params=params)
        x_data = datagen.add_noise_x(const_err=x_err)
        y_data = datagen.add_noise_y(const_err=y_err)

        measurement = Measurement(x=None, y=y_data)

        measurement = Measurement(x=x_data, y=y_data)

    def test_init(self):
        def linear_fun(x, c):
            return c[0] * x + c[1]

        data_len = 10
        boundaries = [1.0, 10.0]
        params = [1, 0]
        x_err = 0.1 * np.ones(data_len)
        y_err = 0.1 * np.ones(data_len)
        datagen = DataGenerator(linear_fun, data_len, boundaries, params=params)
        x_data = datagen.add_noise_x(const_err=x_err)
        y_data = datagen.add_noise_y(const_err=y_err)

        measurement = Measurement()

        with self.assertRaises(TypeError) as context:
            measurement = Measurement(x=x_data, y=y_data)
            measurement.set_data(x=None, y=y_data)

        with self.assertRaises(TypeError) as context:
            measurement = Measurement(x=5, y=y_data)

        with self.assertRaises(TypeError) as context:
            measurement = Measurement(x=[0, 1], y=[0, 1, 2])

        with self.assertRaises(TypeError) as context:
            measurement = Measurement(x=[[0, 0], [1, 1]], y=[0, 1], x_err=[0.1, 0.1])
