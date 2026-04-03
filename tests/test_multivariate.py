"""Tests for multivariable (2-D X) input across all three regressors."""
import unittest

import numpy as np

from mcup._utils import local_numpy_seed
from mcup.deming import DemingRegressor
from mcup.weighted import WeightedRegressor
from mcup.xy_weighted import XYWeightedRegressor

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def bilinear(x, p):
    """y = p[0]*x[0] + p[1]*x[1] + p[2]"""
    return p[0] * x[0] + p[1] * x[1] + p[2]


def linear_x0(x, p):
    """y = p[0] + p[1]*x[0]  (only first feature matters)"""
    return p[0] + p[1] * x[0]


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

N = 40
TRUE_BILINEAR = np.array([2.0, -1.0, 0.5])
TRUE_LINEAR = np.array([1.0, 3.0])
P0_BILINEAR = np.zeros(3)
P0_LINEAR = np.zeros(2)

_rng = np.random.default_rng(0)
X2 = np.column_stack([
    np.linspace(0.0, 5.0, N),
    np.linspace(-2.0, 2.0, N),
])
Y_BILINEAR = np.array([bilinear(X2[i], TRUE_BILINEAR) for i in range(N)])
Y_LINEAR = np.array([linear_x0(X2[i], TRUE_LINEAR) for i in range(N)])


# ---------------------------------------------------------------------------
# 1. WeightedRegressor — 2-D X, only y errors
# ---------------------------------------------------------------------------

class TestWeightedRegressor2D(unittest.TestCase):
    def _make_data(self):
        y_err = 0.3 * np.ones(N)
        with local_numpy_seed(1):
            y_noisy = Y_BILINEAR + np.random.normal(0, 0.3, N)
        return X2.copy(), y_noisy, y_err

    def test_analytical_params_close_to_truth(self):
        X, y, y_err = self._make_data()
        est = WeightedRegressor(bilinear, method="analytical")
        est.fit(X, y, y_err=y_err, p0=P0_BILINEAR)
        np.testing.assert_allclose(est.params_, TRUE_BILINEAR, atol=0.5)

    def test_mc_params_close_to_truth(self):
        X, y, y_err = self._make_data()
        with local_numpy_seed(2):
            est = WeightedRegressor(bilinear, method="mc", n_iter=500)
            est.fit(X, y, y_err=y_err, p0=P0_BILINEAR)
        np.testing.assert_allclose(est.params_, TRUE_BILINEAR, atol=0.5)

    def test_analytical_mc_agree(self):
        X, y, y_err = self._make_data()
        est_a = WeightedRegressor(bilinear, method="analytical")
        est_a.fit(X, y, y_err=y_err, p0=P0_BILINEAR)
        with local_numpy_seed(3):
            est_mc = WeightedRegressor(bilinear, method="mc", n_iter=800)
            est_mc.fit(X, y, y_err=y_err, p0=P0_BILINEAR)
        np.testing.assert_allclose(est_a.params_, est_mc.params_, atol=0.3)

    def test_covariance_shape_and_finite(self):
        X, y, y_err = self._make_data()
        est = WeightedRegressor(bilinear, method="analytical")
        est.fit(X, y, y_err=y_err, p0=P0_BILINEAR)
        self.assertEqual(est.covariance_.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(est.covariance_)))
        self.assertTrue(np.all(est.params_std_ > 0))


# ---------------------------------------------------------------------------
# 2. XYWeightedRegressor — 2-D X, second column x_err=0 (controlled variable)
# ---------------------------------------------------------------------------

class TestXYWeightedRegressor2DZeroErrColumn(unittest.TestCase):
    def _make_data(self):
        # First feature: measured with error.  Second feature: exact.
        x_err = np.column_stack([0.2 * np.ones(N), np.zeros(N)])
        y_err = 0.3 * np.ones(N)
        with local_numpy_seed(4):
            X_noisy = X2 + np.random.normal(0, 1, X2.shape) * x_err
            y_noisy = Y_LINEAR + np.random.normal(0, 0.3, N)
        return X_noisy, y_noisy, x_err, y_err

    def test_analytical_params_close_to_truth(self):
        X, y, x_err, y_err = self._make_data()
        est = XYWeightedRegressor(linear_x0, method="analytical")
        est.fit(X, y, x_err=x_err, y_err=y_err, p0=P0_LINEAR)
        np.testing.assert_allclose(est.params_, TRUE_LINEAR, atol=0.6)

    def test_mc_params_close_to_truth(self):
        X, y, x_err, y_err = self._make_data()
        with local_numpy_seed(5):
            est = XYWeightedRegressor(linear_x0, method="mc", n_iter=500)
            est.fit(X, y, x_err=x_err, y_err=y_err, p0=P0_LINEAR)
        np.testing.assert_allclose(est.params_, TRUE_LINEAR, atol=0.6)

    def test_mc_zero_err_column_receives_no_noise(self):
        """Zero-error features must not be perturbed during MC sampling."""
        X, y, x_err, y_err = self._make_data()
        # Intercept the x_s values inside mc_solve by wrapping cost_fn_builder
        sampled_second_cols = []

        original_fit = XYWeightedRegressor.fit

        def patched_fit(self_est, X_in, y_in, x_err_in, y_err_in, p0):
            from mcup import _mc as mc_mod
            original_mc_solve = mc_mod.mc_solve

            def recording_mc_solve(cost_fn_builder, X_, y_, x_err_, y_err_, p0_, *args, **kwargs):
                def wrapped_builder(x_s, y_s, params_est):
                    sampled_second_cols.append(x_s[:, 1].copy())
                    return cost_fn_builder(x_s, y_s, params_est)
                return original_mc_solve(
                    wrapped_builder, X_, y_, x_err_, y_err_, p0_, *args, **kwargs
                )

            mc_mod.mc_solve = recording_mc_solve
            try:
                return original_fit(self_est, X_in, y_in, x_err_in, y_err_in, p0)
            finally:
                mc_mod.mc_solve = original_mc_solve

        with local_numpy_seed(6):
            est = XYWeightedRegressor(linear_x0, method="mc", n_iter=20)
            patched_fit(est, X, y, x_err, y_err, P0_LINEAR)

        # Every sampled second column must equal the original (zero-error column)
        for col in sampled_second_cols:
            np.testing.assert_array_almost_equal(col, X[:, 1], decimal=10)


# ---------------------------------------------------------------------------
# 3. DemingRegressor — 2-D X, second column x_err=0
# ---------------------------------------------------------------------------

class TestDemingRegressor2DZeroErrColumn(unittest.TestCase):
    def _make_data(self):
        x_err = np.column_stack([0.3 * np.ones(N), np.zeros(N)])
        y_err = 0.3 * np.ones(N)
        with local_numpy_seed(6):
            X_noisy = X2 + np.random.normal(0, 1, X2.shape) * x_err
            y_noisy = Y_LINEAR + np.random.normal(0, 0.3, N)
        return X_noisy, y_noisy, x_err, y_err

    def test_analytical_no_division_by_zero(self):
        X, y, x_err, y_err = self._make_data()
        est = DemingRegressor(linear_x0, method="analytical")
        est.fit(X, y, x_err=x_err, y_err=y_err, p0=P0_LINEAR)
        self.assertTrue(np.all(np.isfinite(est.params_)))
        self.assertTrue(np.all(np.isfinite(est.covariance_)))

    def test_analytical_params_close_to_truth(self):
        X, y, x_err, y_err = self._make_data()
        est = DemingRegressor(linear_x0, method="analytical")
        est.fit(X, y, x_err=x_err, y_err=y_err, p0=P0_LINEAR)
        np.testing.assert_allclose(est.params_, TRUE_LINEAR, atol=0.6)

    def test_mc_no_division_by_zero(self):
        X, y, x_err, y_err = self._make_data()
        with local_numpy_seed(7):
            est = DemingRegressor(linear_x0, method="mc", n_iter=200)
            est.fit(X, y, x_err=x_err, y_err=y_err, p0=P0_LINEAR)
        self.assertTrue(np.all(np.isfinite(est.params_)))

    def test_mc_params_close_to_truth(self):
        X, y, x_err, y_err = self._make_data()
        with local_numpy_seed(8):
            est = DemingRegressor(linear_x0, method="mc", n_iter=200)
            est.fit(X, y, x_err=x_err, y_err=y_err, p0=P0_LINEAR)
        np.testing.assert_allclose(est.params_, TRUE_LINEAR, atol=0.6)


# ---------------------------------------------------------------------------
# 4. DemingRegressor — all x_err zero (degenerates to y-only fit)
# ---------------------------------------------------------------------------

class TestDemingRegressorAllZeroXErr(unittest.TestCase):
    def _make_data(self):
        x_err = np.zeros_like(X2)
        y_err = 0.3 * np.ones(N)
        with local_numpy_seed(9):
            y_noisy = Y_LINEAR + np.random.normal(0, 0.3, N)
        return X2.copy(), y_noisy, x_err, y_err

    def test_analytical_all_zero_x_err_no_crash(self):
        X, y, x_err, y_err = self._make_data()
        est = DemingRegressor(linear_x0, method="analytical")
        est.fit(X, y, x_err=x_err, y_err=y_err, p0=P0_LINEAR)
        self.assertTrue(np.all(np.isfinite(est.params_)))

    def test_mc_all_zero_x_err_no_crash(self):
        X, y, x_err, y_err = self._make_data()
        with local_numpy_seed(10):
            est = DemingRegressor(linear_x0, method="mc", n_iter=100)
            est.fit(X, y, x_err=x_err, y_err=y_err, p0=P0_LINEAR)
        self.assertTrue(np.all(np.isfinite(est.params_)))


if __name__ == "__main__":
    unittest.main()
