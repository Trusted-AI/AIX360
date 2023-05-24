import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod
from aix360.algorithms.tsutils.tsframe import tsFrame


class TSFeatures(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplemented("Error: method name() not implemented in base class!")

    @abstractmethod
    def feat_compute(
        self,
        x: Union[np.ndarray, "tsFrame"],
    ):
        raise NotImplemented(
            "Error: method feat_compute() not implemented in base class!"
        )

    def batch_compute(
        self,
        xx: List[Union[np.ndarray, "tsFrame"]],
    ):
        return [self.feat_compute(x) for x in xx]

    @property
    @abstractmethod
    def range(self):
        raise NotImplemented
