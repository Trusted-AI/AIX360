import numpy as np
from typing import Union
from aix360.algorithms.tsutils.tsframe import tsFrame, to_np_array
from aix360.algorithms.tsutils.tsfeatures.tsfeatures import TSFeatures

SUPPORTED_STATS = [
    "median",
    "mean",
    "min",
    "max",
    "std",
    "range",
    "intercept",
    "trend",
    "rsquared",
    "max_variation",
]


class RangeFeature(TSFeatures):
    """RangeFeature is an instance of TSFeature. It computes the data statistics over
    any specified contiguous temporal scope specified by the time range. The length of
    the temporal scope is specified explicitly by the length argument, and the start
    point of the temporal scope is specified by the start argument during the object
    instantiation. Various statistics supported by the current implementations are:
    1. mean/average
    2. min/minimum.
    3. max/maximum
    4. std/standard deviation,
    5. range/value range,
    6. intercept/intercept for linear regression fit,
    7. trend / slope of the linear regression fit,
    8. rsquared / r2 or squared pearson correlation
    9. max_deviation / unsigned maximum change between successive observations.
    """

    def __init__(
        self,
        start: int,
        length: int,
        stat: str,
    ):
        if stat.lower() not in SUPPORTED_STATS:
            raise ValueError(f"Error: unsupported statistics {stat}!")
        super(RangeFeature, self).__init__()
        self._name = f"{stat}"
        self._start = start
        self._range = length
        self._stat = stat.lower()

    @property
    def name(self) -> str:
        return self._name

    @property
    def range(self):
        return self._start, self._start + self._range

    def feat_compute(
        self,
        x: Union[np.ndarray, "tsFrame"],
    ):
        if x.shape[0] <= self._start:
            raise ValueError(f"Error: expects atleast {self._start + 1} observations!")

        start_pt = self._start
        end = self._start + self._range

        xs = to_np_array(x)[start_pt:end]
        n = xs.shape[0]

        if self._stat == "mean":
            return np.nanmean(xs, axis=0)
        elif self._stat == "median":
            return np.nanmedian(xs, axis=0)
        elif self._stat == "min":
            return np.nanmin(xs, axis=0)
        elif self._stat == "max":
            return np.nanmax(xs, axis=0)
        elif self._stat == "std":
            return np.nanstd(xs, axis=0, ddof=1)
        elif self._stat == "trend":
            return np.asarray(
                [np.polyfit(np.arange(n), xs[:, i], 1)[0] for i in range(xs.shape[1])]
            )
        elif self._stat == "intercept":
            return np.asarray(
                [np.polyfit(np.arange(n), xs[:, i], 1)[1] for i in range(xs.shape[1])]
            )
        elif self._stat == "rsquared":
            return np.asarray(
                [
                    np.corrcoef(xs[:, i], np.arange(n, dtype=float))[0, 1] ** 2
                    for i in range(xs.shape[1])
                ]
            )
        elif self._stat == "range":
            return np.nanmax(xs, axis=0) - np.nanmin(xs, axis=0)
        elif self._stat == "max_variation":
            if n > 1:
                dx = np.diff(xs, axis=0)
                return np.array(
                    [dx[j, i] for i, j in enumerate(np.argmax(np.abs(dx), axis=0))]
                )
            else:
                return np.zeros(xs.shape[1])
        else:
            raise Exception(
                "Selected feature extraction: {} is not implemented.".format(self._stat)
            )
