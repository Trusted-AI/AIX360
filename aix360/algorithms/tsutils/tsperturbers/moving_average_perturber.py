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


class MovingAveragePerturber(TSPerturber):
    """MovingAveragePerturber maintains the moving mean of the time series
    data as computed using specified window length, but add perturbed noise
    with similar distribution as the data with MA structure.
    """

    def __init__(
        self,
        window_length: int = 5,
        lag: int = 5,
        block_length: int = 5,
    ):
        """
        MovingAveragePerturber initialization

        Args:
            window_length (int): window length for mean and residual estimation. Defaults to 5.
            lag (int): lag parameter for MA noise generation. Defaults to 5.
            block_length (int): length of the contiguous time window (block) to perturb. Defaults to 5.
        """
        super(MovingAveragePerturber, self).__init__()
        self._mean = None
        self._residual = None
        self._data_length = None
        self._parameters = dict()
        self._parameters["window_length"] = window_length
        self._parameters["lag"] = lag
        self._parameters["block_length"] = block_length

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
        lag = self._parameters.get("lag")
        block_length = self._parameters.get("block_length")

        x_res = self._residual.copy()
        if isinstance(x_res, pd.DataFrame):
            x_res = x_res.values.T
        else:
            x_res = x_res.T

        params = np.asarray(
            [
                np.corrcoef(
                    np.asarray([np.roll(x, shift=-i) for i in range(lag)])[:, :-lag]
                )[0]
                for x in x_res
            ]
        )
        ranges = np.asarray([[np.min(x), np.max(x)] for x in x_res]) / np.sqrt(lag)
        n_res = []
        f = 1.0 / np.sqrt(lag)
        for i in range(n_perturbations):
            r_seq = []
            for j, _ in enumerate(x_res):
                f_r = (
                    f
                    * (2 * np.random.random(self._data_length) - 1)
                    * (ranges[j, 1] - ranges[j, 0])
                )
                x_r = np.convolve(f_r, params[j], "same")
                r_seq.append(x_r)
            r_seq = np.asarray(r_seq).T
            if isinstance(self._mean, pd.DataFrame):
                r_seq = pd.DataFrame(
                    r_seq, index=self._mean.index, columns=self._mean.columns
                )
            n_res.append(r_seq)

        if block_selector is not None:
            from_point = block_selector.select_start_point(
                x=self._residual, n=n_perturbations
            )
            mask = np.zeros((n_perturbations,) + self._residual.shape)
            for i, pos in enumerate(from_point):
                mask[i, pos : (pos + block_length)] = 1.0

            perturbed_instances = []
            for i, res in enumerate(n_res):
                perturbed_instances.append(
                    self._mean + mask[i] * res + (1 - mask[i]) * self._residual
                )
            return perturbed_instances

        return [self._mean + res for res in n_res]
