import json

import numpy as np
from numpy.testing import assert_allclose

import pandas as pd
from pandas.util.testing import assert_frame_equal

import unittest

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, max_error

from aix360.algorithms.rbm import FeatureBinarizer, GLRMExplainer, LinearRuleRegression


class TestLinearRuleRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boston = load_boston()

    def test_classification(self):
        boston_df = pd.DataFrame(self.boston.data, columns=self.boston.feature_names)
        X_train, X_test, Y_train, Y_test = train_test_split(boston_df, self.boston.target, test_size = 0.25, random_state = 31)
        fb = FeatureBinarizer(negations=True)
        X_train_fb = fb.fit_transform(X_train)
        X_test_fb = fb.transform(X_test)

        self.assertEqual(len(X_train_fb.columns), 196)
        self.assertEqual(len(X_test_fb.columns), 196)

        linear_model = LinearRuleRegression()
        explainer = GLRMExplainer(linear_model)
        explainer.fit(X_train_fb, Y_train)
        Y_pred = explainer.predict(X_test_fb)

        self.assertGreater(r2_score(Y_test, Y_pred), 0.8)
        self.assertGreater(explained_variance_score(Y_test, Y_pred), 0.8)
        self.assertLess(mean_absolute_error(Y_test, Y_pred), 3)
        self.assertLess(max_error(Y_test, Y_pred), 12)

        explanation = explainer.explain()
        explanation = explainer.explain()
        expected = pd.DataFrame(columns=["rule", "coefficient"], data=[
            ['(intercept)', 21.9],
            ['NOX <= 0.66', 6.3],
            ['RM <= 7.16 AND DIS > 1.62', -5.8],
            ['LSTAT <= 4.66', 5.5],
            ['DIS <= 3.32 AND RAD > 2.00 AND B > 295.98 AND LSTAT <= 22.79', 4.8],
            ['CHAS == 0.0 AND PTRATIO > 16.10', -3.9],
            ['RM <= 7.16 AND RAD <= 6.00', -3.3],
            ['TAX > 293.00 AND LSTAT > 4.66', -2.9],
            ['LSTAT <= 15.03', 2.8],
            ['INDUS > 4.05 AND LSTAT > 4.66', -2.5],
            ['DIS <= 7.24 AND RAD > 2.00 AND PTRATIO <= 20.90 AND B <= 394.99 AND B > 295.98 AND LSTAT <= 22.79', 2.5],
            ['LSTAT <= 9.48', 2.5],
            ['CRIM <= 9.84 AND DIS <= 4.64 AND RAD > 1.00 AND TAX <= 666.00 AND LSTAT <= 22.79', 2.2],
            ['LSTAT <= 17.60', 1.9],
            ['TAX > 330.00 AND LSTAT > 4.66', -1.8],
            ['CRIM <= 9.84 AND CRIM > 0.06 AND PTRATIO <= 20.90', 1.8],
            ['LSTAT <= 6.25', 1.6],
            ['RM <= 7.16 AND B > 380.27', -1.6],
            ['LSTAT <= 11.12', 1.6],
            ['RAD > 2.00 AND LSTAT <= 22.79', 1.2],
            ['RM <= 7.16', -1.2],
            ['CHAS == 0.0 AND RM <= 7.16', 1.2],
            ['RM <= 6.51', -1.1],
            ['CRIM <= 9.84 AND DIS <= 3.95 AND TAX <= 666.00 AND PTRATIO <= 20.90 AND B > 295.98', 1.0],
            ['CRIM <= 9.84 AND RAD > 1.00 AND LSTAT <= 22.79', 1.0],
            ['DIS <= 3.95 AND LSTAT <= 22.79', -0.9],
            ['RM <= 6.74', -0.8],
            ['PTRATIO <= 19.52', 0.8],
            ['NOX <= 0.66 AND PTRATIO <= 20.90 AND LSTAT <= 22.79', -0.8],
            ['RAD > 4.00 AND LSTAT <= 22.79', -0.63],
            ['B <= 391.27 AND LSTAT <= 22.79', 0.5],
            ['LSTAT <= 7.58', 0.44],
            ['LSTAT <= 13.14', 0.17]
        ])
        assert_frame_equal(explanation, expected, check_dtype=False, check_exact=False, check_less_precise=1)

        figs, _ = explainer.visualize(boston_df, fb)
        with open('tests/rbm/linear_plot_data.json') as fp:
            plot_data = json.load(fp)
            for k,v in plot_data.items():
                obtained_plot = figs[k].axes[0].lines[0].get_xydata()
                assert_allclose(np.array(v), obtained_plot, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
