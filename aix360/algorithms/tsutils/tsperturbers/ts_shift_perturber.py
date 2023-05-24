import numpy as np
import pandas as pd
from typing import Union
from scipy.interpolate import interp1d
from aix360.algorithms.tsutils.tsframe import tsFrame, to_np_array
from aix360.algorithms.tsutils.tsperturbers.tsperturber import (
    BlockSelector,
    TSPerturber,
)


class TSShiftPerturber(TSPerturber):
    """TSShiftPerturber adds random lag in a time continuous block of the
    time series data. The lag introduction cause gap in the data, which is
    imputed using the specified interpolation function.
    """

    def __init__(
        self,
        max_shift: int = 2,
        block_length: int = 5,
        n_blocks: int = 1,
        interpolation_kind: str = "linear",
    ):
        """TSShiftPerturber initialization

        Args:
            max_shift (int): maximum allowed lag. Defaults to 2.
            block_length (int): block size lag is introduced in selected blocks. Defaults to 5.
            n_blocks (int): number of blocks to perturb. Defaults to 1.
            interpolation_kind (str): interpolation method for data imputation. Refer to
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html.
                Defaults to "linear".
        """
        super(TSShiftPerturber, self).__init__()
        self._data = None
        self._parameters = dict()
        self._parameters["block_length"] = block_length
        self._parameters["max_shift"] = max_shift
        self._parameters["n_blocks"] = n_blocks
        self._parameters["interpolation_kind"] = interpolation_kind

    def get_params(self):
        return self._parameters.copy()

    def set_params(self, **kwargs):
        self._parameters.update(kwargs)
        return self

    def _fit(
        self,
        x: Union["tsFrame", np.ndarray],
    ):
        self._data = x.copy()
        return self

    def _transform(
        self,
        n_perturbations: int = 1,
        block_selector: BlockSelector = None,
    ):
        n_blocks = self._parameters.get("n_blocks")
        max_shift = self._parameters.get("max_shift")
        block_length = self._parameters.get("block_length")
        interpolation_kind = self._parameters.get("interpolation_kind")

        data = to_np_array(self._data)
        x_data = np.arange(data.shape[0]).astype("float32")
        length, dim = data.shape

        perturbed_data = []
        margin = length - max(max_shift, block_length)

        for i in range(n_perturbations):
            p_data = np.zeros_like(data)
            for j in range(dim):
                y_data = data[:, j]
                px_data = x_data.copy()

                if block_selector is None:
                    blocks = np.random.randint(max_shift, margin, n_blocks)
                else:
                    blocks = block_selector.select_start_point(
                        x=self._data, n=n_blocks, margin=margin
                    )
                for b in blocks:
                    shift = (2 * np.random.random() - 1) * max_shift
                    px_data[b : (b + block_length)] = (
                        x_data[b : (b + block_length)] + shift
                    )

                ix = np.argsort(px_data)
                poly = interp1d(
                    px_data[ix], y_data[ix], kind=interpolation_kind, copy=True
                )
                p_data[:, j] = poly(x_data)
            if isinstance(self._data, pd.DataFrame):
                p_data = pd.DataFrame(
                    p_data, index=self._data.index, columns=self._data.columns
                )
            perturbed_data.append(p_data)
        return perturbed_data
