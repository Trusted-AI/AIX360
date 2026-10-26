"""Exercise the public RBM estimators with real sklearn solvers."""
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from aix360.algorithms.rbm import (
    FeatureBinarizer, LinearRuleRegression, LogisticRuleRegression,
)


class TestRuleRegressionFitReturn(unittest.TestCase):
    def setUp(self):
        # Use the real public transformer to produce the supported column schema.
        values = np.tile([0, 1], 20)
        raw = pd.DataFrame({"flag": values})
        self.binarizer = FeatureBinarizer(negations=False)
        self.X = self.binarizer.fit_transform(raw)
        self.assertIsInstance(self.X.columns, pd.MultiIndex)
        self.assertEqual(self.X.columns.nlevels, 3)
        assert_array_equal(self.X.iloc[:, 0].to_numpy(), values)
        self.labels = values.copy()
        self.responses = 2.0 + 3.0 * values

    def logistic_model(self, debias=False):
        return LogisticRuleRegression(
            lambda0=0.01, lambda1=0.001, iterMax=0,
            debias=debias, maxSolverIter=2000,
        )

    def linear_model(self, debias=False):
        return LinearRuleRegression(
            lambda0=0.01, lambda1=0.001, iterMax=0, debias=debias,
        )

    def test_logistic_fit_returns_same_estimator(self):
        for debias in (False, True):
            with self.subTest(debias=debias):
                model = self.logistic_model(debias)
                result = model.fit(self.X, self.labels)
                self.assertIs(result, model)
                self.assertTrue(hasattr(result, "lr"))
                self.assertTrue(hasattr(result, "z"))

    def test_linear_fit_returns_same_estimator(self):
        for debias in (False, True):
            with self.subTest(debias=debias):
                model = self.linear_model(debias)
                result = model.fit(self.X, self.responses)
                self.assertIs(result, model)
                self.assertTrue(hasattr(result, "lr"))
                self.assertTrue(hasattr(result, "z"))

    def test_logistic_fit_predict_chain(self):
        model = self.logistic_model()
        prediction = model.fit(self.X, self.labels).predict(self.X)
        assert_array_equal(prediction, self.labels)
        assert_array_equal(prediction, model.predict(self.X))

    def test_linear_fit_predict_chain(self):
        model = self.linear_model()
        prediction = model.fit(self.X, self.responses).predict(self.X)
        # Regularization slightly shrinks the fitted contrast; no exact solver
        # coefficients or timing are part of the return-value contract.
        assert_allclose(prediction, self.responses, atol=0.1, rtol=0)
        assert_array_equal(prediction, model.predict(self.X))

    def test_logistic_existing_split_workflow(self):
        model = self.logistic_model(debias=True)
        model.fit(self.X, self.labels)
        assert_array_equal(model.predict(self.X), self.labels)
        probabilities = model.predict_proba(self.X)
        self.assertEqual(probabilities.shape, self.labels.shape)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((0 <= probabilities) & (probabilities <= 1)).all())
        self.assertFalse(model.explain().empty)

    def test_linear_existing_split_workflow(self):
        model = self.linear_model(debias=True)
        model.fit(self.X, self.responses)
        assert_allclose(model.predict(self.X), self.responses, atol=0.1, rtol=0)
        self.assertFalse(model.explain().empty)


if __name__ == "__main__":
    unittest.main()
