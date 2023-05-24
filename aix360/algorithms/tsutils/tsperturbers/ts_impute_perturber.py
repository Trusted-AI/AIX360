import numpy as np
import pandas as pd
from typing import Union
from scipy.interpolate import interp1d
from aix360.algorithms.tsutils.tsframe import tsFrame, to_np_array
from aix360.algorithms.tsutils.tsperturbers.tsperturber import (
    BlockSelector,
    TSPerturber,
)


class TSImputePerturber(TSPerturber):
    """TSImputePerturber removes random block from the time series data, and imputes
    the value with specified interpolation method.
    """

    def __init__(
        self,
        block_length: int = 5,
        n_blocks: int = 1,
        sparsity: float = 1.0,
        padding: int = 5,
        interpolation_kind: str = "linear",
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    ):
        """TSImputePerturber initialization

        Args:
            block_length (int): length of the block size, continuous time window length over
                which perturbation will be performed. Defaults to 5.
            n_blocks (int): number of blocks to perturb. Defaults to 1.
            sparsity (float): sparsity controls data sampling, there by creating across data
                sparse sampling, Defaults to 1.0.
            padding (int): this parameter is used smoothen the imputation boundaries. Defaults to 5.
            interpolation_kind (str): interpolation method for data imputation. Refer to
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
                for possible values. Defaults to linear.
        """
        super(TSImputePerturber, self).__init__()
        self._data = None
        self._parameters = dict()
        self._parameters["block_length"] = block_length
        self._parameters["n_blocks"] = n_blocks
        self._parameters["sparsity"] = sparsity
        self._parameters["padding"] = padding
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
        padding = self._parameters.get("padding")
        n_blocks = self._parameters.get("n_blocks")
        sparsity = self._parameters.get("sparsity")
        block_length = self._parameters.get("block_length")
        interpolation_kind = self._parameters.get("interpolation_kind")

        data = to_np_array(self._data)
        x_data = np.arange(data.shape[0])
        length, dim = data.shape

        perturbed_data = []

        for i in range(n_perturbations):
            p_data = np.zeros_like(data)
            for j in range(dim):
                y_data = data[:, j]
                margin = length - block_length - padding
                if block_selector is None:
                    blocks = np.random.randint(padding, margin, n_blocks)
                else:
                    blocks = block_selector.select_start_point(
                        x=self._data, n=n_blocks, margin=margin
                    )
                block = []
                for b in blocks:
                    block = block + list(range(b, b + block_length))
                block = np.unique(block)
                index = list(set(x_data).difference(block))
                n = int((len(index) - 2 * padding) * sparsity)
                index = (
                    index[:padding]
                    + sorted(
                        np.random.choice(index[padding:-padding], n, replace=False)
                    )
                    + index[-padding:]
                )
                poly = interp1d(
                    x_data[index], y_data[index], kind=interpolation_kind, copy=True
                )
                p_data[:, j] = poly(x_data)
            if isinstance(self._data, pd.DataFrame):
                p_data = pd.DataFrame(
                    p_data, index=self._data.index, columns=self._data.columns
                )
            perturbed_data.append(p_data)
        return perturbed_data
