import warnings
import numpy as np
import pandas as pd
from typing import List, Union, Callable
from aix360.algorithms.tslbbe import TSLocalBBExplainer
from aix360.algorithms.tsutils.tsframe import tsFrame, to_np_array
from aix360.algorithms.tsutils.tsperturbers.tsperturber import (
    TSPerturber,
    BlockSelector,
)
from aix360.algorithms.tsutils.tsperturbers.perturbed_data_generator import (
    PerturbedDataGenerator,
)
from aix360.algorithms.tsutils.tsfeatures.latest_features import LatestFeature
from aix360.algorithms.tsutils.tsfeatures.range_features import RangeFeature


class TSICEExplainer(TSLocalBBExplainer):
    """TSICEExplainer extends the Individual Conditional Expectation for correlated
    timeseries data (higher dimensions). It uses TSFeatures to derive time series
    structural features, and uses data perturber(TSPerturber) for generating simulated
    data. TSICEExplainer explains the trend in the model forecast change with time series
    derived features.

    References:
        .. [#0] `Goldstein et al. 'Peeking Inside the Black Box: Visualizing Statistical
            Learning with Plots of Individual Conditional Expectation'
            <https://arxiv.org/abs/1309.6392>`_
    """

    def __init__(
        self,
        forecaster: Callable,
        input_length: int,
        forecast_lookahead: int,
        n_variables: int = 1,
        n_exogs: int = 0,
        n_perturbations: int = 25,
        features_to_analyze: List[str] = None,
        perturbers: List[Union[TSPerturber, dict]] = None,
        explanation_window_start: int = None,
        explanation_window_length: int = 10,
    ):
        """Initializer for TSICEExplainer

        Args:
            forecaster (Callable): Callable object produces a forecast as numpy array
                for a given input as numpy array.
            input_length (int): Input length for the forecaster.
            forecast_lookahead (int): Lookahead length of the forecaster prediction.
            n_variables (int): Number of variables in the forecaster input. Defaults to 1.
            n_exogs (int): Number of exogenous variable required for the forecaster. Defaults to 0.
            n_perturbations (int): Number of perturbed instance for TSExplanation. Defaults to 25.
            features_to_analyze (List[str]): List of features used to analyze the perturbed timeseries
                during TSICE explanation. As the observing timeseries is complicated, these
                set of features can be used to observe the perturbations closer. Allowed
                values are "median", "mean", "min", "max", "std", "range", "intercept",
                "trend", "rsquared", "max_variation". If None, "mean" is used by default.
                Defaults to None.
            perturbers (List[TSPerturber, dict]): data perturbation algorithm specification by TSPerturber
                instance or dict. Allowed values for "type" key in dictionary are block-bootstrap, frequency,
                moving-average, shift. Block-bootstrap split the time series into contiguous
                chunks called blocks, for each block noise is estimated and noise is exchanged
                and added to the signal between randomly selected blocks. Moving-average perturbation
                maintains the moving mean of the time series data with the specified window length,
                but add perturbed noise with similar distribution as the data. Frequency
                perturber performs FFT on the noise, and removes random high frequency
                components from the noise estimates. Number of frequencies to be removed
                is specified by the truncate_frequencies argument. Shift perturber adds
                random upward or downward shift in the data value over time continuous
                blocks. If not provided default perturber is combination of block-bootstrap,
                moving-average, and frequency. Defaults to None.
            explanation_window_start (int): Explanation window is selected from the input timeseries
                starting explanation_window_start. This window is used to select the part of the timeseries
                for TSICE analysis. Perturbations are computed over this explanation window. If
                explanation_window_start is None, explanation_window is selected from recent upto
                explanation_window_length. Defaults to None.
            explanation_window_length (int): Explanation window of length: explanation_window_length is
                selected from the input timeseries. This window is used to select the part of the timeseries
                for TSICE analysis. Perturbations are computed over this explanation window. Defaults to 10.
        """
        self.model = forecaster
        if features_to_analyze is None:
            features_to_analyze = ["mean"]

        feature_class = LatestFeature
        feature_class_args = {}
        if explanation_window_length > input_length:
            raise Exception(
                "explanation_window_length must be positive integer and lesser than or equal to input_length"
            )

        if explanation_window_start is None:  # recent window
            self.explanation_window_start = -explanation_window_length
            self.explanation_window_length = None
            feature_class = LatestFeature
            feature_class_args["length"] = explanation_window_length
        else:  # range
            if explanation_window_start >= input_length:
                raise Exception(
                    "explanation_window_start must be positive integer and lesser than input_length"
                )
            self.explanation_window_start = explanation_window_start
            self.explanation_window_length = (
                explanation_window_start + explanation_window_length
            )
            feature_class = RangeFeature
            feature_class_args["start"] = explanation_window_start
            feature_class_args["length"] = explanation_window_length

        self.feat_extractors = []

        for stat in features_to_analyze:
            feature_class_args["stat"] = stat
            self.feat_extractors.append(feature_class(**feature_class_args))

        block_selector = BlockSelector(
            start=self.explanation_window_start, end=self.explanation_window_length
        )

        if perturbers is None:
            perturbers = [
                dict(type="block-bootstrap"),
            ]

        self._parameters = {
            "input_length": input_length,
            "forecast_lookahead": forecast_lookahead,
            "n_variables": n_variables,
            "n_exogs": n_exogs,
            "n_perturbations": n_perturbations,
            "pertubers": perturbers,
            "explanation_window_start": explanation_window_start,
            "explanation_window_length": explanation_window_length,
        }

        self.perturber = PerturbedDataGenerator(
            perturber_engines=perturbers,
            block_selector=block_selector,
        )
        self.tensor_model = False
        super(TSICEExplainer, self).__init__()
        self._is_fitted = True

    def get_params(self):
        return self._parameters.copy()

    def set_params(self, *argv, **kwargs):
        self._parameters.update(kwargs)
        return self

    def _explain_instance(
        self,
        ts: Union["tsFrame", np.ndarray],
        ts_related: Union["tsFrame", np.ndarray],
        **explain_params,
    ):

        x = ts
        x_exog = ts_related
        input_length = explain_params.get(
            "input_length", self._parameters.get("input_length")
        )
        lookahead = explain_params.get(
            "forecast_lookahead", self._parameters.get("forecast_lookahead")
        )
        n_variables = explain_params.get(
            "n_variables", self._parameters.get("n_variables")
        )
        n_exogs = explain_params.get("n_exogs", self._parameters.get("n_exogs"))
        n_perturbations = explain_params.get(
            "n_perturbations", self._parameters.get("n_perturbations")
        )

        n_obs, n_vars = x.shape
        assert (
            n_obs >= input_length
        ), f"Error: model requires min input length {input_length}!"
        assert (
            n_vars == n_variables
        ), f"Error: expects {n_variables} endogenous variable(s) but found {n_vars}!"

        if n_exogs > 0:
            assert (
                x_exog is not None
            ), f"Error: expects {n_exogs} exogenous features to be specific!"
            assert x_exog.shape[1] == n_exogs, (
                f"Error: exogenous dimension does not match "
                f"{n_exogs} != {x_exog.shape[1]}"
            )
            assert (
                x_exog.shape[0] == x.shape[0]
            ), f"Error: exogenous and endogenous variable are not aligned!"
            x_exog = x_exog[-(input_length + lookahead) :]

        x = x[-input_length:]

        if self.tensor_model:
            xc = to_np_array(x)
            if x_exog is not None:
                xc_exog = to_np_array(x_exog)
                forecast = np.asarray(
                    self.model([xc.reshape(1, -1), xc_exog.reshape(1, -1)])[0]
                )
            else:
                forecast = np.asarray(self.model(xc.reshape(1, -1))[0])
        else:
            if x_exog is not None:
                try:
                    forecast = np.asarray(self.model(x, x_exog))
                except:
                    xc = to_np_array(x)
                    xc_exog = to_np_array(x_exog)
                    forecast = np.asarray(
                        self.model([xc.reshape(1, -1), xc_exog.reshape(1, -1)])[0]
                    )
                    self.tensor_model = True
            else:
                try:
                    forecast = np.asarray(self.model(x))
                except:
                    xc = to_np_array(x)
                    forecast = np.asarray(self.model(xc.reshape(1, -1))[0])
                    self.tensor_model = True
        if len(forecast.shape) == 1:
            forecast = np.reshape(forecast, (-1, 1))
        assert forecast.shape[0] == lookahead
        assert forecast.shape[1] == n_variables

        x_data, _ = self.perturber.fit_transform(x, None, n_perturbations)

        feature_sets = []
        feat_names = []
        current_features_set = []
        for feat_extractor in self.feat_extractors:
            features = np.asarray(feat_extractor.batch_compute(x_data))
            feature_sets.append(features.tolist())
            feat_names.append(feat_extractor.name)
            current_features_set.append(np.asarray(feat_extractor.feat_compute(x)))

        base_f = forecast

        signed_impact = []
        unsigned_impact = []
        forecasts_on_perturbations = []
        for i in range(n_perturbations):
            if self.tensor_model:
                xc_data = to_np_array(x_data[i])
                if x_exog is not None:
                    f = np.asarray(
                        self.model([xc_data.reshape(1, -1), xc_exog.reshape(1, -1)])[0]
                    )
                else:
                    f = np.asarray(self.model(xc_data.reshape(1, -1))[0])
            else:
                if x_exog is not None:
                    f = np.asarray(self.model(x_data[i], x_exog))
                else:
                    f = np.asarray(self.model(x_data[i]))
            if len(f.shape) == 1:
                f = np.reshape(f, (-1, 1))

            forecasts_on_perturbations.append(f)
            sif = np.mean(np.mean(f - base_f, axis=0))
            uif = np.mean(np.sqrt(np.mean(np.square(f - base_f), axis=0)))
            signed_impact.append(sif)
            unsigned_impact.append(uif)

        signed_impact = np.asarray(signed_impact)
        unsigned_impact = np.asarray(unsigned_impact)

        if isinstance(ts, pd.DataFrame):
            data_x = ts[-signed_impact.shape[0] :].to_dict()
        else:
            data_x = ts[-signed_impact.shape[0] :].tolist()

        explanation = {}
        explanation["data_x"] = data_x
        explanation["current_forecast"] = base_f
        explanation["feature_names"] = feat_names
        explanation["feature_values"] = feature_sets
        explanation["signed_impact"] = signed_impact.tolist()
        explanation["total_impact"] = unsigned_impact.tolist()
        explanation["current_feature_values"] = current_features_set
        explanation["perturbations"] = [pert.to_dict() for pert in x_data]
        explanation["forecasts_on_perturbations"] = forecasts_on_perturbations

        return explanation

    def explain_instance(
        self, ts: tsFrame, ts_related: tsFrame = None, **explain_params
    ):
        """Explain the forecast made by the forecaster at a certain point in time
        (**local explanation**).

        Args:
            ts (tsFrame): The future univariate time series ``tsFrame`` to use for forecasting that
                extends forward from the end of training ``tsFrame`` (``ts_train``) or
                ``timestamp_start`` for the requested number of periods.
                This can be generated using :py:mod:`aix360.algorithms.tsframe.tsFrame`.
                A ``tsFrame`` is a pandas ``DataFrame`` indexed by ``Timestamp`` objects
                (that is ``DatetimeIndex``). Each column corresponds to a target to forecast.
            ts_related (tsFrame, optional): The related time series ``tsFrame`` containing
                the external regressors. A ``tsFrame`` is a pandas ``DataFrame`` indexed by
                ``Timestamp`` objects (that is ``DatetimeIndex``). Each column corresponds to a
                related external regressor. Defaults to None.
            explain_params: Arbitrary explainer parameters.

        Returns:
            dict: explanation object
                Dictionary with data_x, feature_names, feature_values, signed_impact, total_impact,
                current_forecast, current_feature_values, perturbations and forecasts_on_perturbations.

        """

        return super(TSICEExplainer, self).explain_instance(
            ts=ts, ts_related=ts_related, **explain_params
        )
