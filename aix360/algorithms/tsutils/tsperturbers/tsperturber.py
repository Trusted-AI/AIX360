import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod


class BlockSelector:
    """
    BlockSelector is used to prepare the effective window to compute explanation
    from the user provided parameters. This is used while computing timeseries perturbations.
    """

    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end

    def select_start_point(
        self, x, n: int = 1, margin: int = None, block_length: int = 5
    ):
        start, end = self._start, self._end
        x = np.array(x)
        t = x.shape[0]

        if (margin is not None) and (margin < 0):
            margin = t - margin

        if margin is None:
            margin = t

        if margin < 0:
            raise ValueError(
                f"Error: margin should be a valid point with in data length!"
            )

        if start < 0:
            start = t + start
            if start < 0:
                start = 0

        if (end is not None) and (end < 0):
            end = t + end
            if end < 0:
                raise ValueError(f"Error: end must be within the index range!")
        elif end is None:
            end = t

        end = min(end, margin)

        return np.random.randint(low=start, high=end, size=n)


class TSPerturber(ABC):
    """Abstract interface for time series perturbation."""

    def __init__(self):
        self._fitted = False

    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        x: Union[pd.DataFrame, np.ndarray],
    ):
        try:
            self._fit(x)
            self._fitted = True
        except RuntimeError as e:
            raise e
        return self

    def fit_transform(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        n_perturbations: int = 1,
        block_selector: BlockSelector = None,
    ):
        self.fit(x)
        return self.transform(
            n_perturbations=n_perturbations, block_selector=block_selector
        )

    def transform(
        self,
        n_perturbations: int = 1,
        block_selector: BlockSelector = None,
    ):
        if not self.is_fitted():
            raise RuntimeError(
                "Error: transform must be called after fitting the data!"
            )
        return self._transform(n_perturbations, block_selector)

    @abstractmethod
    def _fit(self, x: Union[pd.DataFrame, np.ndarray]):
        pass

    @abstractmethod
    def _transform(
        self, n_perturbations: int = 1, block_selector: BlockSelector = None
    ):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        pass
