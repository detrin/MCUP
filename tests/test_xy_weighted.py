import unittest
import numpy as np
from mcup._utils import local_numpy_seed
from mcup.data_generator import DataGenerator


def linear(x, p):
    return p[0] + p[1] * x


class TestXYWeightedRegressor(unittest.TestCase):
    def setUp(self):
        self.n = 30
        self.true_params = np.array([1.0, 2.0])
        self.p0 = np.array([0.0, 0.0])
        dg = DataGenerator(linear, self.n, [0.0, 10.0], params=self.true_params)
        self.X = dg.x
        self.y_err = 0.5 * np.ones(self.n)
        self.x_err = 0.3 * np.ones(self.n)
        with local_numpy_seed(42):
            self.y = dg.add_noise_y(const_err=self.y_err)
            self.X_noisy = dg.add_noise_x(const_err=self.x_err)

    def test_analytical_params_close_to_truth(self):
        from mcup.xy_weighted import XYWeightedRegressor
        est = XYWeightedRegressor(linear, method="analytical")
        est.fit(self.X_noisy, self.y, x_err=self.x_err, y_err=self.y_err, p0=self.p0)
        np.testing.assert_allclose(est.params_, self.true_params, atol=0.5)

    def test_analytical_covariance_shape(self):
        from mcup.xy_weighted import XYWeightedRegressor
        est = XYWeightedRegressor(linear, method="analytical")
        est.fit(self.X_noisy, self.y, x_err=self.x_err, y_err=self.y_err, p0=self.p0)
        self.assertEqual(est.covariance_.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(est.covariance_)))

    def test_mc_params_close_to_truth(self):
        from mcup.xy_weighted import XYWeightedRegressor
        with local_numpy_seed(42):
            est = XYWeightedRegressor(linear, method="mc", n_iter=500)
            est.fit(self.X_noisy, self.y, x_err=self.x_err, y_err=self.y_err, p0=self.p0)
        np.testing.assert_allclose(est.params_, self.true_params, atol=0.5)

    def test_params_std_positive(self):
        from mcup.xy_weighted import XYWeightedRegressor
        with local_numpy_seed(42):
            est = XYWeightedRegressor(linear, method="mc", n_iter=200)
            est.fit(self.X_noisy, self.y, x_err=self.x_err, y_err=self.y_err, p0=self.p0)
        self.assertTrue(np.all(est.params_std_ > 0))

    def test_not_fitted_raises(self):
        from mcup.xy_weighted import XYWeightedRegressor
        est = XYWeightedRegressor(linear)
        with self.assertRaises(ValueError):
            est._check_is_fitted()


if __name__ == "__main__":
    unittest.main()
