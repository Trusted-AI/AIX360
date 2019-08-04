import json

import numpy as np
from numpy.testing import assert_allclose

import pandas as pd
from pandas.util.testing import assert_frame_equal

import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from aix360.algorithms.rbm import FeatureBinarizer, GLRMExplainer, LogisticRuleRegression


class TestLogisticRuleRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_breast_cancer()

    def test_classification(self):
        bc_df = pd.DataFrame(self.bc.data, columns=self.bc.feature_names)
        X_train, X_test, Y_train, Y_test = train_test_split(bc_df, self.bc.target, test_size = 0.2, random_state = 31)
        fb = FeatureBinarizer(negations=True)
        X_train_fb = fb.fit_transform(X_train)
        X_test_fb = fb.transform(X_test)

        self.assertEqual(len(X_train_fb.columns), 540)
        self.assertEqual(len(X_test_fb.columns), 540)

        logistic_model = LogisticRuleRegression(maxSolverIter=2000)
        explainer = GLRMExplainer(logistic_model)
        explainer.fit(X_train_fb, Y_train)
        Y_pred = explainer.predict(X_test_fb)

        self.assertGreater(accuracy_score(Y_test, Y_pred), 0.85)
        self.assertGreater(precision_score(Y_test, Y_pred), 0.85)
        self.assertGreater(recall_score(Y_test, Y_pred), 0.85)
        self.assertGreater(f1_score(Y_test, Y_pred), 0.9)

        explanation = explainer.explain()
        expected = pd.DataFrame(columns=["rule", "coefficient"], data=[
          ['(intercept)', -11.2],
          ['worst perimeter <= 116.46 AND worst concave points <= 0.15', -11.9],
          ['worst concave points <= 0.15', 10.1],
          ['worst perimeter <= 116.46 AND worst concave points <= 0.18', 9.8],
          ['worst area <= 930.88', 5.4],
          ['worst area > 680.60 AND worst concavity > 0.22', -3.3],
          ['worst perimeter <= 116.46 AND worst smoothness <= 0.16', 3.1],
          ['mean concave points <= 0.05', 1.5],
          ['worst concavity <= 0.27', 0.9],
          ['worst concave points <= 0.12', 0.63],
          ['worst perimeter <= 104.38', -0.02]
        ])
        assert_frame_equal(explanation, expected, check_dtype=False, check_exact=False, check_less_precise=1)

        figs, _ = explainer.visualize(bc_df, fb)
        with open('tests/rbm/logistic_plot_data.json') as fp:
            plot_data = json.load(fp)
            for k,v in plot_data.items():
                obtained_plot = figs[k].axes[0].lines[0].get_xydata()
                assert_allclose(np.array(v), obtained_plot, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
