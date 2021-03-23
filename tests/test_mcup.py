import unittest
import numpy as np

from mcup import fun


class TestMCUP(unittest.TestCase):
    def test_fun(self):
        y = fun(3, 1, 0)
        self.assertEqual(y, 3)

        y = fun(3, 0, 1)
        self.assertEqual(y, 1)

        y = fun(3 * np.ones((10)), 1, 0)
        self.assertTrue(np.allclose(y, 3 * np.ones((10))))

        y = fun(3 * np.ones((10)), 0, 1)
        self.assertTrue(np.allclose(y, 1 * np.ones((10))))
