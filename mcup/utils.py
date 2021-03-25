"""
utils.py
====================================
Utilities for the package.
"""

import contextlib
import numpy as np


@contextlib.contextmanager
def local_numpy_seed(seed):
    """Set temporal seed for numpy package with local scope.

    Args:
        seed ([int]): Seed for numpy package.
    """
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            np.random.set_state(state)
