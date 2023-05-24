import numpy as np
from collections import Counter
from typing import List, Union
from aix360.algorithms.tsutils.tsframe import tsFrame
from aix360.algorithms.tsutils.tsperturbers import (
    BlockBootstrapPerturber,
    FrequencyPerturber,
    MovingAveragePerturber,
    TSShiftPerturber,
    TSImputePerturber,
)
from aix360.algorithms.tsutils.tsperturbers.tsperturber import (
    BlockSelector,
    TSPerturber,
)


class PerturbedDataGenerator:
    """
    PerturbedDataGenerator is a wrapping class to prepare various kinds of
    perturbers and generate specified number of perturbations using these
    perturbers.
    """

    def __init__(
        self,
        perturber_engines: List[Union[TSPerturber, dict]] = None,
        block_selector: BlockSelector = None,
    ):
        """
        Constructor method, initializes the explainer

        Args:
            perturber_engines (List[TSPerturber, dict]): data perturbation algorithm specification
                by TSPerturber instance or dict. Allowed values for "type" key in dictionary are
                block-bootstrap, frequency, moving-average, shift. Block-bootstrap split the time series
                into contiguous chunks called blocks, for each block noise is estimated and noise is exchanged
                and added to the signal between randomly selected blocks. Moving-average perturbation
                maintains the moving mean of the time series data with the specified window length,
                but add perturbed noise with similar distribution as the data. Frequency
                perturber performs FFT on the noise, and removes random high frequency
                components from the noise estimates. Number of frequencies to be removed
                is specified by the truncate_frequencies argument. Shift perturber adds
                random upward or downward shift in the data value over time continuous
                blocks. If not provided default perturber is combination of block-bootstrap,
                moving-average, and frequency. Default: None
            block_selector (BlockSelector): The block_selector is used to prepare the effective window to
                compute explanation from the user provided parameters. This is used while computing timeseries
                perturbations.
        """
        self._perturbers = []
        self._block_selector = block_selector

        if (perturber_engines is None) or (len(perturber_engines) == 0):
            perturber_engines = [
                dict(type="block-bootstrap"),
                dict(type="moving_average"),
                dict(type="frequency"),
            ]

        for engine in perturber_engines:
            if isinstance(engine, TSPerturber):
                self._perturbers.append(engine)
            elif isinstance(engine, dict):
                assert all([f in engine for f in ["type"]])
                if engine.get("type") == "block-bootstrap":
                    self._perturbers.append(
                        BlockBootstrapPerturber(
                            window_length=engine.get("window_length", 5),
                            block_length=engine.get("block_length", 5),
                            block_swap=engine.get("block_swaps", 2),
                        )
                    )
                elif engine.get("type") == "frequency":
                    self._perturbers.append(
                        FrequencyPerturber(
                            window_length=engine.get("window_length", 5),
                            truncate_frequencies=engine.get("truncate_frequencies", 4),
                            block_length=engine.get("block_length", 5),
                        )
                    )
                elif engine.get("type") == "moving-average":
                    self._perturbers.append(
                        MovingAveragePerturber(
                            window_length=engine.get("window_length", 5),
                            lag=engine.get("lag", 5),
                            block_length=engine.get("block_length", 5),
                        )
                    )
                elif engine.get("type") == "shift":
                    self._perturbers.append(
                        TSShiftPerturber(
                            max_shift=engine.get("max_shift", 2),
                            block_length=engine.get("block_length", 5),
                            n_blocks=engine.get("n_blocks", 1),
                            interpolation_kind=engine.get(
                                "interpolation_kind", "linear"
                            ),
                        )
                    )
                elif engine.get("type") == "impute":
                    self._perturbers.append(
                        TSImputePerturber(
                            block_length=engine.get("block_length", 5),
                            n_blocks=engine.get("n_blocks", 1),
                            sparsity=engine.get("sparsity", 1.0),
                            padding=engine.get("padding", 5),
                            interpolation_kind=engine.get(
                                "interpolation_kind", "linear"
                            ),
                        )
                    )

        if len(self._perturbers) == 0:
            raise RuntimeError(f"Error: no valid perturber specified!")

    def fit_transform(
        self,
        x: Union["tsFrame", np.ndarray],
        x_exog: Union["tsFrame", np.ndarray] = None,
        n: int = 10,
    ):
        counter = Counter(np.random.choice(len(self._perturbers), n))
        data = []
        data_exog = []
        for idx in counter:
            ni = counter.get(idx)
            data += self._perturbers[idx].fit_transform(
                x, n_perturbations=ni, block_selector=self._block_selector
            )
            if x_exog is not None:
                data_exog += self._perturbers[idx].fit_transform(
                    x_exog, n_perturbations=ni
                )
        return data, data_exog
