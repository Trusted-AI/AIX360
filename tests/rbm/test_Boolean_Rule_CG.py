import pandas as pd
import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from aix360.algorithms.rbm import FeatureBinarizer, BRCGExplainer, BooleanRuleCG


class TestBooleanmRuleCG(unittest.TestCase):
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

        boolean_model = BooleanRuleCG(silent=True)
        explainer = BRCGExplainer(boolean_model)
        explainer.fit(X_train_fb, Y_train)
        Y_pred = explainer.predict(X_test_fb)

        self.assertGreater(accuracy_score(Y_test, Y_pred), 0.9)
        self.assertGreater(precision_score(Y_test, Y_pred), 0.9)
        self.assertGreater(recall_score(Y_test, Y_pred), 0.9)
        self.assertGreater(f1_score(Y_test, Y_pred), 0.9)

        explanation = explainer.explain()
        self.assertEqual(explanation['rules'], [
          'compactness error > 0.01 AND worst concavity <= 0.22 AND worst symmetry <= 0.28',
          'mean texture <= 15.46 AND mean concavity <= 0.15 AND area error <= 54.16',
          'fractal dimension error > 0.00 AND worst area <= 680.60 AND worst concave points <= 0.18',
          'mean concave points <= 0.05 AND perimeter error <= 3.80 AND worst area <= 930.88 AND worst smoothness <= 0.16'
        ])


if __name__ == '__main__':
    unittest.main()
