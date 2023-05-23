""" Tests for :py:mod:`aix360.algorithms.tsice.TSICEExplainer`.
"""
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from aix360.algorithms.tsutils.tsframe import tsFrame
from aix360.datasets import SunspotDataset
from aix360.algorithms.tsice import TSICEExplainer
from aix360.algorithms.tsutils.tsperturbers import BlockBootstrapPerturber

# transform a time series dataset into a supervised learning dataset
# below sample forecaster is from: https://machinelearningmastery.com/random-forest-for-time-series-forecasting/
class RandomForestUniVariateForecaster:
    def __init__(self, n_past=4, n_future=1, RFparams={"n_estimators": 250}):
        self.n_past = n_past
        self.n_future = n_future
        self.model = RandomForestRegressor(**RFparams)

    def fit(self, X):
        train = self._series_to_supervised(X, n_in=self.n_past, n_out=self.n_future)
        trainX, trainy = train[:, : -self.n_future], train[:, -self.n_future :]
        self.model = self.model.fit(trainX, trainy)
        return self

    def _series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols = list()

        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        # put it all together
        agg = pd.concat(cols, axis=1)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg.values

    def predict(self, X):
        row = X[-self.n_past :].flatten()
        y_pred = self.model.predict(np.asarray([row]))
        return y_pred


class TestTSICEExplainer(unittest.TestCase):
    def setUp(self):

        # load data
        df, schema = SunspotDataset().load_data()
        ts = tsFrame(
            df, timestamp_column=schema["timestamp"], columns=schema["targets"]
        )

        (self.ts_train, self.ts_test) = train_test_split(
            ts, shuffle=False, stratify=None, test_size=0.15, train_size=None
        )

    def test_tsice_with_range(self):

        # load model
        input_length = 24
        forecast_horizon = 4
        forecaster = RandomForestUniVariateForecaster(
            n_past=input_length, n_future=forecast_horizon
        )

        forecaster.fit(self.ts_train.iloc[-200:])

        # initialize/fit explainer
        observation_length = 12
        explainer = TSICEExplainer(
            forecaster=forecaster.predict,
            explanation_window_start=10,
            explanation_window_length=observation_length,
            features_to_analyze=[
                "mean",  # analyze mean metric from recent time series of lengh <observation_length>
            ],
            perturbers=[
                BlockBootstrapPerturber(window_length=5, block_length=5, block_swap=2),
            ],
            input_length=input_length,
            forecast_lookahead=forecast_horizon,
            n_perturbations=20,
        )

        # compute explanations
        explanation = explainer.explain_instance(
            ts=self.ts_test.iloc[:80],
        )

        # validate explanation structure
        self.assertIn("data_x", explanation)
        self.assertIn("feature_names", explanation)
        self.assertIn("feature_values", explanation)
        self.assertIn("signed_impact", explanation)
        self.assertIn("total_impact", explanation)
        self.assertIn("current_forecast", explanation)
        self.assertIn("current_feature_values", explanation)
        self.assertIn("perturbations", explanation)
        self.assertIn("forecasts_on_perturbations", explanation)

    def test_tsice_with_latest(self):

        # load model
        input_length = 24
        forecast_horizon = 4
        forecaster = RandomForestUniVariateForecaster(
            n_past=input_length, n_future=forecast_horizon
        )

        forecaster.fit(self.ts_train.iloc[-200:])

        # initialize/fit explainer
        observation_length = 12
        explainer = TSICEExplainer(
            forecaster=forecaster.predict,
            explanation_window_start=None,
            explanation_window_length=observation_length,
            features_to_analyze=[
                "mean",  # analyze mean metric from recent time series of lengh <observation_length>
                "median",  # analyze median metric from recent time series of lengh <observation_length>
                "std",  # analyze std metric from recent time series of lengh <observation_length>
                "max_variation",  # analyze max_variation metric from recent time series of lengh <observation_length>
                "min",
                "max",
                "range",
                "intercept",
                "trend",
                "rsquared",
            ],
            perturbers=[
                BlockBootstrapPerturber(window_length=5, block_length=5, block_swap=2),
                dict(
                    type="frequency",
                    window_length=5,
                    truncate_frequencies=5,
                    block_length=4,
                ),
                dict(type="moving-average", window_length=5, lag=5, block_length=4),
                dict(type="impute", block_length=4),
                dict(type="shift", block_length=4),
            ],
            input_length=input_length,
            forecast_lookahead=forecast_horizon,
            n_perturbations=20,
        )

        # compute explanations
        explanation = explainer.explain_instance(
            ts=self.ts_test.iloc[:80],
        )

        # validate explanation structure
        self.assertIn("data_x", explanation)
        self.assertIn("feature_names", explanation)
        self.assertIn("feature_values", explanation)
        self.assertIn("signed_impact", explanation)
        self.assertIn("total_impact", explanation)
        self.assertIn("current_forecast", explanation)
        self.assertIn("current_feature_values", explanation)
        self.assertIn("perturbations", explanation)
        self.assertIn("forecasts_on_perturbations", explanation)


if __name__ == "__main__":
    unittest.main()
