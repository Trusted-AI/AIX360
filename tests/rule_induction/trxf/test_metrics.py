from unittest import TestCase

import numpy as np
import pandas as pd

import aix360.algorithms.rule_induction.trxf.metrics as metrics
from tests.rule_induction.trxf.utilities import create_numerical_test_data, create_numerical_test_ruleset_pos


class TestMetrics(TestCase):
    def test_compute_ruleset_metrics(self):
        X, y = create_numerical_test_data(10)
        ruleset = create_numerical_test_ruleset_pos()
        actual = metrics.compute_ruleset_metrics(ruleset, X, y)
        expected = {'tn': 3, 'fp': 4, 'fn': 0, 'tp': 3, 'accuracy': 0.6}
        self.assertEqual(actual, expected)

    def test_compute_rule_metrics(self):
        X, y = create_numerical_test_data(10)
        ruleset = create_numerical_test_ruleset_pos()
        actual = metrics.compute_rule_metrics(ruleset, X, y)
        expected = [
            metrics.RuleContributionMetrics(ruleset.conjunctions[0], 1, 0, 0, 1, 7, 10, 0.5),
            metrics.RuleContributionMetrics(ruleset.conjunctions[1], 2, 1, 1, 2, 7, 10, 0.5),
            metrics.RuleContributionMetrics(ruleset.conjunctions[2], 2, 0, 1, 0, 8, 10, 0.6),
            metrics.RuleContributionMetrics(ruleset.conjunctions[3], 1, 0, 0, 0, 8, 10, 2/3)
        ]
        self.assertEqual(actual, expected)

    def test_get_preaggregated_confusion_matrix(self):
        pos_value = 1
        y_p = pd.Series([0, 1, 1, 0, 1])
        y_t = pd.Series([0, 0, 1, 1, 1])
        actual = metrics.get_preaggregated_confusion_matrix(y_p, y_t, pos_value)
        expected = (np.array([2, 4]), np.array([0]), np.array([1]), np.array([3]))
        np.testing.assert_equal(actual, expected)

    def test_ruleset_complexity(self):
        ruleset = create_numerical_test_ruleset_pos()
        actual = metrics.compute_ruleset_complexity(ruleset)
        expected = metrics.RuleComplexityMetrics(4, 4, 10, 4)
        self.assertEqual(actual, expected)
