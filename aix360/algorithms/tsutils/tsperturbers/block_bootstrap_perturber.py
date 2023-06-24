import numpy as np
import pandas as pd
from typing import Union
from aix360.algorithms.tsutils.tsframe import tsFrame
from aix360.algorithms.tsutils.tsperturbers.tsperturber import (
    BlockSelector,
    TSPerturber,
)
from aix360.algorithms.tsutils.tsperturbers.perturber_utils import (
    ts_split_mean_residual,
)


class BlockBootstrapPerturber(TSPerturber):
    """BlockBootstrapPerturber split the time series into contiguous chunks
    called blocks, for each block noise is estimated and noise is exchanged
    and added to the signal (mean) between randomly selected blocks.

    References:
        .. [#0] `Bühlmann, Peter. "Bootstraps for time series."
            Statistical science (2002): 52-72.
            <https://projecteuclid.org/journals/statistical-science/volume-17/issue-1/Bootstraps-for-Time-Series/10.1214/ss/1023798998.full>`_
    """

    def __init__(
        self, window_length: int = 5, block_length: int = 5, block_swap: int = 2
    ):
        """BlockBootstrapPerturber initialization.

        Args:
            window_length (int): window length used for noise estimation. Defaults to 5.
            block_length (int): block length, perturber swaps noise between blocks. Defaults to 5.
            block_swap (int): number of block pairs for perturbation. Defaults to 2.
        """
        super(BlockBootstrapPerturber, self).__init__()
        self._mean = None
        self._residual = None
        self._data_length = None
        self._parameters = dict()
        self._parameters["window_length"] = window_length
        self._parameters["block_length"] = block_length
        self._parameters["block_swap"] = block_swap

    def get_params(self):
        return self._parameters.copy()

    def set_params(self, **kwargs):
        self._parameters.update(kwargs)
        return self

    def _fit(
        self,
        x: Union["tsFrame", np.ndarray],
    ):
        window_length = self._parameters.get("window_length")
        self._mean, self._residual = ts_split_mean_residual(
            x, window_size=window_length
        )
        self._data_length = x.shape[0]
        return self

    def _transform(
        self,
        n_perturbations: int = 1,
        block_selector: BlockSelector = None,
    ):
        block_length = self._parameters.get("block_length")
        block_swap = self._parameters.get("block_swap")

        x_res = [self._residual.copy() for _ in range(n_perturbations)]
        margin = self._residual.shape[0] - block_length + 1
        for _ in range(block_swap):
            if block_selector is None:
                from_point = np.random.randint(
                    0, self._data_length - block_length, n_perturbations
                )

                to_point = np.random.randint(
                    0, self._data_length - block_length, n_perturbations
                )
            else:
                from_point = block_selector.select_start_point(
                    x=self._residual, n=n_perturbations, margin=margin
                )
                to_point = block_selector.select_start_point(
                    x=self._residual, n=n_perturbations, margin=margin
                )

            for j, start in enumerate(zip(from_point, to_point)):
                start_1, start_2 = start
                x_res[j][start_1 : (start_1 + block_length)] = self._residual[
                    start_2 : (start_2 + block_length)
                ]
                x_res[j][start_2 : (start_2 + block_length)] = self._residual[
                    start_1 : (start_1 + block_length)
                ]

        return [self._mean + res for res in x_res]
