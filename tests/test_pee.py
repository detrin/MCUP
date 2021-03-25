import unittest
import numpy as np

from mcup import parameter_error_estimator
from mcup import DataGenerator
from mcup.utils import local_numpy_seed


class TestPEE(unittest.TestCase):
    def test_pee(self):
        def linear_fun(x, c):
            return c[0] * x + c[1]

        with local_numpy_seed(42):
            data_len = 10
            boundaries = [1.0, 10.0]
            params = [1, 0]
            params_0 = [0, 0]
            y_err = 0.1 * np.ones(data_len)
            x_err = np.zeros_like(y_err)
            datagen = DataGenerator(linear_fun, data_len, boundaries, params=params)
            x_data = datagen.x
            y_data = datagen.add_noise_y(const_err=y_err)

            with self.assertRaises(TypeError) as context:
                params_mean, params_std = parameter_error_estimator(
                    linear_fun,
                    x_data,
                    y_data,
                    x_err,
                    y_err,
                    params_0,
                    method="Nelder-Mead",
                )

            with self.assertRaises(TypeError) as context:
                params_mean, params_std = parameter_error_estimator(
                    linear_fun,
                    x_data,
                    y_data,
                    x_err,
                    y_err,
                    params_0,
                    rtol=1e-4,
                    method="Nelder-Mead",
                )

            # test for fixed num of iterations
            params_mean, params_std = parameter_error_estimator(
                linear_fun,
                x_data,
                y_data,
                x_err,
                y_err,
                params_0,
                iter_num=10,
                method="Nelder-Mead",
            )

            self.assertTrue(
                np.allclose(params_mean, np.array([0.99999637, 0.04765171]))
            )

            self.assertTrue(np.allclose(params_std, np.array([0.01035922, 0.04579523])))

            # test for relative convergence
            params_mean, params_std = parameter_error_estimator(
                linear_fun,
                x_data,
                y_data,
                x_err,
                y_err,
                params_0,
                rtol=1e-4,
                atol=1e-4,
                method="Nelder-Mead",
            )

            self.assertTrue(
                np.allclose(params_mean, np.array([0.99934391, 0.04822805]))
            )
            self.assertTrue(np.allclose(params_std, np.array([0.01042134, 0.06693809])))

            # test for fixed num of iterations
            params_mean, params_std = parameter_error_estimator(
                linear_fun,
                x_data,
                y_data,
                x_err,
                y_err,
                params_0,
                iter_num=10,
                method="Newton-CG",
            )

            # self.assertTrue(
            #     np.allclose(params_mean, np.array([1.00181933, 0.03373715]))
            # )
            # self.assertTrue(np.allclose(params_std, np.array([0.01167075, 0.06066596])))

            # test for relative convergence

            params_mean, params_std = parameter_error_estimator(
                linear_fun,
                x_data,
                y_data,
                x_err,
                y_err,
                params_0,
                rtol=1e-4,
                atol=1e-4,
                method="Newton-CG",
            )

            # self.assertTrue(
            #     np.allclose(params_mean, np.array([0.99943959, 0.05364996]))
            # )
            # self.assertTrue(np.allclose(params_std, np.array([0.00896757, 0.05064698])))
