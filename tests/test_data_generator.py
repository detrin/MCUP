import unittest
import numpy as np

from mcup import DataGenerator
from mcup.utils import local_numpy_seed


class TestDataGenerator(unittest.TestCase):
    def test_init(self):
        def fun(x):
            return x

        data_len = 5
        boundaries = [[1.0, 5.0]]
        with self.assertRaises(TypeError) as context:
            DataGenerator(None, data_len, boundaries)

        with self.assertRaises(TypeError) as context:
            DataGenerator(fun, None, boundaries)

        with self.assertRaises(TypeError) as context:
            DataGenerator(fun, data_len, None)

        with self.assertRaises(TypeError) as context:
            DataGenerator(fun, data_len, [[0.0, 1.0, 2.0]])

        with self.assertRaises(TypeError) as context:
            DataGenerator(fun, data_len, [[[0.0, 1.0, 2.0]]])

        with self.assertRaises(TypeError) as context:
            DataGenerator(fun, data_len, [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

        with self.assertRaises(TypeError) as context:
            DataGenerator(fun, data_len, [[0.0, 1.0], [1.0, 0.0]])

        with self.assertRaises(TypeError) as context:
            DataGenerator(fun, data_len, [[1.0, 0.0]])

        datagen = DataGenerator(fun, data_len, boundaries)

        self.assertTrue(np.allclose(datagen.x, np.array([1.0, 2.0, 3.0, 4.0, 5.0])))
        self.assertTrue(np.allclose(datagen.y, np.array([1.0, 2.0, 3.0, 4.0, 5.0])))

        def fun2(x):
            return x[0] + x[1]

        data_len = 2
        boundaries = [[1.0, 2.0], [1.0, 2.0]]
        datagen = DataGenerator(fun2, data_len, boundaries)

        self.assertTrue(np.allclose(datagen.x, np.array([[1.0, 1.0], [2.0, 2.0]])))
        self.assertTrue(np.allclose(datagen.y, np.array([2.0, 4.0])))

    def test_add_noise(self):
        def fun(x):
            return x

        data_len = 5
        const_err = [0.1, 0.1, 0.2, 0.1, 0.1]
        stat_err = [0.0, 0.01, 0.1, 1.0, 10.0]
        boundaries = [[1.0, 5.0]]

        datagen = DataGenerator(fun, data_len, boundaries, seed=42)

        with self.assertRaises(AssertionError) as context:
            datagen._DataGenerator__add_noise(
                datagen.x, const_err=None, stat_error=None
            )
        with local_numpy_seed(42):
            data = datagen._DataGenerator__add_noise(datagen.x, const_err=const_err)
            data_ref = np.array(
                [1.04967142, 1.98617357, 3.12953771, 4.15230299, 4.97658466]
            )
            self.assertTrue(np.allclose(data, data_ref))

        with local_numpy_seed(42):
            data = datagen._DataGenerator__add_noise(datagen.x, stat_error=stat_err)
            data_ref = np.array([1.0, 1.99723471, 3.19430656, 10.09211943, -6.70766874])
            self.assertTrue(np.allclose(data, data_ref))

        with local_numpy_seed(42):
            data = datagen._DataGenerator__add_noise(
                datagen.x, const_err=const_err, stat_error=stat_err
            )
            data_ref = np.array(
                [0.9765863, 2.155156, 3.34779351, 10.04517199, -6.65341273]
            )
            self.assertTrue(np.allclose(data, data_ref))

    def test_add_noise_x(self):
        def fun(x):
            return x

        data_len = 5
        const_err = [0.1, 0.1, 0.2, 0.1, 0.1]
        stat_err = [0.0, 0.01, 0.1, 1.0, 10.0]
        boundaries = [[1.0, 5.0]]

        datagen = DataGenerator(fun, data_len, boundaries, seed=42)

        data = datagen.add_noise_x(const_err=const_err, stat_error=stat_err)
        data_ref = np.array([0.9765863, 2.155156, 3.34779351, 10.04517199, -6.65341273])

        self.assertTrue(np.allclose(data, data_ref))

    def test_add_noise_y(self):
        def fun(x):
            return x

        data_len = 5
        const_err = [0.1, 0.1, 0.2, 0.1, 0.1]
        stat_err = [0.0, 0.01, 0.1, 1.0, 10.0]
        boundaries = [[1.0, 5.0]]

        datagen = DataGenerator(fun, data_len, boundaries, seed=42)

        data = datagen.add_noise_y(const_err=const_err, stat_error=stat_err)
        data_ref = np.array([0.9765863, 2.155156, 3.34779351, 10.04517199, -6.65341273])

        self.assertTrue(np.allclose(data, data_ref))
