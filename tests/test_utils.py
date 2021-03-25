import unittest
import numpy as np

from mcup.utils import local_numpy_seed


class TestUtils(unittest.TestCase):
    def test_temp_seed(self):
        state = np.random.get_state()
        np.random.seed(42)
        a1 = np.random.uniform(size=(10))
        b1 = np.random.uniform(size=(10))

        # Test for [seed]=int
        np.random.seed(42)
        a2 = np.random.uniform(size=(10))
        with local_numpy_seed(12):
            c1 = np.random.uniform(size=(10))
        b2 = np.random.uniform(size=(10))

        self.assertTrue(np.allclose(a1, a2))
        self.assertFalse(np.allclose(b1, c1))
        self.assertTrue(np.allclose(b1, b2))

        # Test for seed=None
        np.random.seed(42)
        a3 = np.random.uniform(size=(10))
        with local_numpy_seed(None):
            c2 = np.random.uniform(size=(10))
        b3 = np.random.uniform(size=(10))

        self.assertTrue(np.allclose(a1, a3))
        self.assertTrue(np.allclose(b1, c2))
        self.assertFalse(np.allclose(b1, b3))
