import os
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from aix360.algorithms.tsutils.tsframe import tsFrame
from aix360.datasets import SunspotDataset
from aix360.algorithms.tslime import TSLimeExplainer
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


class TestTSLimeExplainer(unittest.TestCase):
    def setUp(self):

        # load data
        df, schema = SunspotDataset().load_data()
        ts = tsFrame(
            df, timestamp_column=schema["timestamp"], columns=schema["targets"]
        )

        (self.ts_train, self.ts_test) = train_test_split(
            ts, shuffle=False, stratify=None, test_size=0.15, train_size=None
        )

    def test_tslime(self):

        # load model
        input_length = 24
        forecast_horizon = 4
        forecaster = RandomForestUniVariateForecaster(
            n_past=input_length, n_future=forecast_horizon
        )

        forecaster.fit(self.ts_train.iloc[-200:])

        # initialize/fit explainer

        relevant_history = 12
        explainer = TSLimeExplainer(
            model=forecaster.predict,
            input_length=input_length,
            relevant_history=relevant_history,
            perturbers=[
                BlockBootstrapPerturber(
                    window_length=min(4, input_length - 1), block_length=2, block_swap=2
                ),
            ],
            n_perturbations=10,
            random_seed=22,
        )

        # compute explanations
        test_window = self.ts_test.iloc[:input_length]
        explanation = explainer.explain_instance(
            ts=test_window,
        )

        # validate explanation structure
        self.assertIn("input_data", explanation)
        self.assertIn("history_weights", explanation)
        self.assertIn("x_perturbations", explanation)
        self.assertIn("y_perturbations", explanation)
        self.assertIn("model_prediction", explanation)
        self.assertIn("surrogate_prediction", explanation)

        self.assertEqual(explanation["history_weights"].shape[0], relevant_history)


if __name__ == "__main__":
    unittest.main()
