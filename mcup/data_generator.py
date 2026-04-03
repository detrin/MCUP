from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np

from ._utils import local_numpy_seed


class DataGenerator:
    """Generates synthetic x/y data with optional noise for testing estimators.

    Parameters:
        fun: Function for generating y data.
        data_len: Length of the data.
        boundaries: 1D ``[a, b]`` or 2D ``[[a_1, b_1], ...]`` array defining x intervals.
        seed: Optional seed for the noise generator.
        dtype: NumPy dtype for generated arrays. Default ``np.float64``.
        params: Optional parameter array passed as second argument to ``fun``.
    """

    def __init__(
        self,
        fun: Callable,
        data_len: int,
        boundaries: Union[list, np.ndarray],
        seed: Optional[int] = None,
        dtype: Any = np.float64,
        params: Optional[np.ndarray] = None,
    ) -> None:
        if not callable(fun):
            raise TypeError("Argument fun has to be callable.")

        if not isinstance(data_len, int):
            raise TypeError("Argument data_len has to be integer.")

        if not isinstance(boundaries, (list, np.ndarray)):
            raise TypeError("Argument data_len has to be list or np.ndarray")

        b: np.ndarray = np.array(boundaries) if isinstance(boundaries, list) else boundaries

        if b.ndim != 2 and b.ndim != 1:
            raise TypeError(
                "Argument boundaries has to have exactly dimensionality of two or one."
            )

        if b.ndim == 2 and b.shape[0] == 1:
            b = b[0]

        if b.ndim == 2:
            self.x_dim = b.shape[0]
            if b.shape[1] != 2:
                raise TypeError(
                    "Argument boundaries has to have defined all intervals "
                    "with exactly two numbers."
                )

            for dim_i in range(self.x_dim):
                if b[dim_i][0] >= b[dim_i][1]:
                    raise TypeError("Invalid interval in argument boundaries.")

            self.x = np.linspace(
                b[:, 0],
                b[:, 1],
                data_len,
                dtype=dtype,
                endpoint=True,
            )

        elif b.ndim == 1:
            if b.shape[0] != 2:
                raise TypeError(
                    "Argument boundaries has to have interval with exactly two numbers."
                )

            self.x_dim = 1
            if b[0] > b[1]:
                raise TypeError("Invalid interval in argument boundaries.")

            self.x = np.linspace(
                b[0],
                b[1],
                data_len,
                dtype=dtype,
                endpoint=True,
            )

        self.data_len = data_len
        self.seed = seed
        self.y = np.zeros((data_len), dtype=dtype)
        for i in range(self.data_len):
            if params is None:
                self.y[i] = fun(self.x[i])
            else:
                self.y[i] = fun(self.x[i], params)

    def __add_noise(
        self,
        data: np.ndarray,
        const_err: Optional[float] = None,
        stat_error: Optional[float] = None,
    ) -> np.ndarray:
        assert const_err is not None or stat_error is not None

        if stat_error is None:
            assert const_err is not None
            data_ret: np.ndarray = data + np.random.normal(loc=0.0, scale=const_err)
        elif const_err is None:
            data_ret = np.multiply(data, np.random.normal(loc=1.0, scale=stat_error))
        else:
            data_ret = np.multiply(
                data, np.random.normal(loc=1.0, scale=stat_error)
            ) + np.random.normal(loc=0.0, scale=const_err)

        return data_ret

    def add_noise_x(
        self,
        const_err: Optional[float] = None,
        stat_error: Optional[float] = None,
    ) -> np.ndarray:
        with local_numpy_seed(self.seed):
            return self.__add_noise(self.x, const_err=const_err, stat_error=stat_error)

    def add_noise_y(
        self,
        const_err: Optional[float] = None,
        stat_error: Optional[float] = None,
    ) -> np.ndarray:
        with local_numpy_seed(self.seed):
            return self.__add_noise(self.y, const_err=const_err, stat_error=stat_error)
