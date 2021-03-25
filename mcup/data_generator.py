"""
data_generator.py
====================================
Data generation for testing.
"""

import numpy as np

from .utils import local_numpy_seed


class DataGenerator:
    def __init__(
        self, fun, data_len, boundaries, seed=None, dtype=np.float64, params=None
    ):
        """DataGenerator takes function and generates data with specified data length and boundaries. Dimensionality 
        of x data is given by shape of boundaries.

        Args:
            fun (function): Function for generating y data.
            data_len (int): Length of the data.
            boundaries (array): One dimensional or multidimensional array, where are specified intervals of x_data. 
                For example let x be x = (x_1, x_2, x_3), then the boundaries should be given as 
                [[a_1, b_1], [a_2, b_2], [a_3, b_3]], where [a, b] is an interval.
            seed (int): Optional seed for noise generator.
            dtype (array type): Specify the type of the data that will be generated. Default is np.float64.
        """
        if not callable(fun):
            raise TypeError("Argument fun has to be callable.")

        if not isinstance(data_len, int):
            raise TypeError("Argument data_len has to be integer.")

        if not isinstance(boundaries, (list, np.ndarray)):
            raise TypeError("Argument data_len has to be list or np.ndarray")

        if isinstance(boundaries, list):
            boundaries = np.array(boundaries)

        if boundaries.ndim != 2 and boundaries.ndim != 1:
            raise TypeError(
                "Argument boundaries has to have exactly dimensionality of two or one."
            )

        if boundaries.ndim == 2 and boundaries.shape[0] == 1:
            boundaries = boundaries[0]

        if boundaries.ndim == 2:
            self.x_dim = boundaries.shape[0]
            if boundaries.shape[1] != 2:
                raise TypeError(
                    "Argument boundaries has to have defined all intervals with exactly two numbers."
                )

            for dim_i in range(self.x_dim):
                if boundaries[dim_i][0] >= boundaries[dim_i][1]:
                    raise TypeError("Invalid interval in argument boundaries.")

            self.x = np.linspace(
                boundaries[:, 0],
                boundaries[:, 1],
                data_len,
                dtype=dtype,
                endpoint=True,
            )

        elif boundaries.ndim == 1:
            if boundaries.shape[0] != 2:
                raise TypeError(
                    "Argument boundaries has to have interval with exactly two numbers."
                )

            self.x_dim = 1
            if boundaries[0] > boundaries[1]:
                raise TypeError("Invalid interval in argument boundaries.")

            self.x = np.linspace(
                boundaries[0], boundaries[1], (data_len), dtype=dtype, endpoint=True,
            )

        self.data_len = data_len
        self.seed = seed
        self.y = np.zeros((data_len), dtype=dtype)
        for i in range(self.data_len):
            if params is None:
                self.y[i] = fun(self.x[i])
            else:
                self.y[i] = fun(self.x[i], params)

    def __add_noise(self, data, const_err=None, stat_error=None):
        """[summary]

        Args:
            data ([type]): [description]
            const_err ([type], optional): [description]. Defaults to None.
            stat_error ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        assert const_err is not None or stat_error is not None

        if stat_error is None:
            data_ret = data + np.random.normal(loc=0.0, scale=const_err)
        elif const_err is None:
            # This way we leverage numpy package for checking type of stat_error.
            data_ret = np.multiply(data, np.random.normal(loc=1.0, scale=stat_error))
        else:
            data_ret = np.multiply(
                data, np.random.normal(loc=1.0, scale=stat_error)
            ) + np.random.normal(loc=0.0, scale=const_err)

        return data_ret

    def add_noise_x(self, const_err=None, stat_error=None):
        """[summary]

        Args:
            const_err ([type], optional): [description]. Defaults to None.
            stat_error ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        with local_numpy_seed(self.seed):
            return self.__add_noise(self.x, const_err=const_err, stat_error=stat_error)

    def add_noise_y(self, const_err=None, stat_error=None):
        """[summary]

        Args:
            const_err ([type], optional): [description]. Defaults to None.
            stat_error ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        with local_numpy_seed(self.seed):
            return self.__add_noise(self.y, const_err=const_err, stat_error=stat_error)
