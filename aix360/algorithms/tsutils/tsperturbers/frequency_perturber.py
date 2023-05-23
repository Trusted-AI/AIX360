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


class FrequencyPerturber(TSPerturber):
    """FrequencyPerturber performs FFT on the noise structure of the time series
    data, and removes high frequencies to from the spectra, and reconstruct the
    residual (noise) and added it back to the signal (mean) for generating the
    perturbed instance. Number of frequencies to be removed is specified by the
    truncate_frequencies argument.
    """

    def __init__(
        self,
        window_length: int = 5,
        truncate_frequencies: int = 5,
        block_length: int = 5,
    ):
        """FrequencyPerturber initialization

        Args:
            window_length (int): window length for noise estimation. Defaults to 5.
            truncate_frequencies (int): number of frequencies to truncate. Defaults to 4.
            block_length (int): length of the contiguous time window (block) to perturb. Defaults to 5.
        """
        super(FrequencyPerturber, self).__init__()
        self._mean = None
        self._residual = None
        self._data_length = None
        self._parameters = dict()
        self._parameters["window_length"] = window_length
        self._parameters["truncate_frequencies"] = truncate_frequencies
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
        rem_frequency = self._parameters.get("truncate_frequencies")
        block_length = self._parameters.get("block_length")

        x_res = self._residual.copy()
        if isinstance(x_res, pd.DataFrame):
            x_res = x_res.values.T
        else:
            x_res = x_res.T
        x_res = [np.fft.fft(x) for x in x_res]
        sel_p = np.asarray([np.abs(np.real(x)) for x in x_res])
        sel_p = np.abs(sel_p)
        sel_p = sel_p / np.sum(np.abs(sel_p), axis=1, keepdims=True)

        n_res = []
        for i in range(n_perturbations):
            r_seq = []
            for j, _ in enumerate(x_res):
                f_idx = np.random.choice(
                    np.arange(self._data_length), rem_frequency, p=sel_p[j]
                )
                x_f = x_res[j].copy()
                x_f[f_idx] = complex(0, 0)
                r_seq.append(np.real(np.fft.ifft(x_f)))
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
