import unittest
import numpy as np

from mcup import Measurement
from mcup import DataGenerator
from mcup import parameter_error_estimator
from mcup.utils import local_numpy_seed


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

    def test_set_function(self):
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

        measurement = Measurement(x=x_data, y=y_data)

        with self.assertRaises(TypeError) as context:
            measurement.set_function(5, params)

        with self.assertRaises(TypeError) as context:
            measurement.set_function(linear_fun, 5)

        self.assertTrue(measurement.set_function(linear_fun, params))

    def test_evaluate_params(self):
        def linear_fun(x, c):
            return c[0] * x + c[1]

        with local_numpy_seed(42):
            data_len = 10
            boundaries = [1.0, 10.0]
            params = [1, 0]
            x_err = 0.1 * np.ones((data_len))
            y_err = 0.1 * np.ones((data_len))
            datagen = DataGenerator(linear_fun, data_len, boundaries, params=params)
            x_data = datagen.add_noise_x(const_err=x_err)
            y_data = datagen.add_noise_y(const_err=y_err)

            measurement = Measurement(x=x_data, y=y_data, y_err=y_err, x_err=x_err)

            with self.assertRaises(RuntimeError) as context:
                measurement.evaluate_params()

            measurement.set_function(linear_fun, params)
            params_mean, params_std = measurement.evaluate_params(
                iter_num=10, num_diff=False
            )

            self.assertTrue(
                np.allclose(params_mean, np.array([0.99351374, -0.08966872]))
            )

            self.assertTrue(np.allclose(params_std, np.array([0.01823801, 0.09489745])))

            params_mean, params_std = measurement.evaluate_params(
                iter_num=10, num_diff=True
            )

            # self.assertTrue(
            #     np.allclose(params_mean, np.array([0.99172421, -0.05093282]))
            # )

            # self.assertTrue(np.allclose(params_std, np.array([0.012989, 0.08706437])))
