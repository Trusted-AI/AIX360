from unittest import TestCase

from aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier import RuleSetClassifier, RuleSelectionMethod, \
    ConfidenceMetric, WeightMetric
from tests.rule_induction.trxf import utilities


class TestRuleSetClassifier(TestCase):
    def test_predict_first_hit(self):
        ruleset = utilities.create_test_ruleset()
        classifier = RuleSetClassifier([ruleset],
                                       rule_selection_method=RuleSelectionMethod.FIRST_HIT,
                                       default_label=0)
        assignment1 = {'toto0': 0.05, 'toto1': 'bar', 'toto2': False, 'toto3': -2}
        actual = classifier.predict(assignment1)
        expected = 1
        self.assertEqual(expected, actual)

        assignment2 = {'toto0': -5, 'toto1': 5, 'toto2': 2, 'toto3': -2}
        actual = classifier.predict(assignment2)
        expected = 0
        self.assertEqual(expected, actual)

    def test_predict_weighted_sum(self):
        classifier = setup_multi_ruleset_classifier(RuleSelectionMethod.WEIGHTED_SUM)
        assignment = {'x1': 2.3, 'x2': -1, 'x3': -3, 'x4': 4}
        actual = classifier.predict(assignment)
        expected = 0
        self.assertEqual(expected, actual)

    def test_predict_weighted_max(self):
        classifier = setup_multi_ruleset_classifier(RuleSelectionMethod.WEIGHTED_MAX)
        assignment = {'x1': 2.3, 'x2': -1, 'x3': -3, 'x4': 4}
        actual = classifier.predict(assignment)
        expected = 0
        self.assertEqual(expected, actual)

    def test_update_rules_with_metrics(self):
        classifier = setup_multi_ruleset_classifier(RuleSelectionMethod.WEIGHTED_MAX)
        expected = [(1, 0.5), (1, 5 / 9), (1, 0.375), (1, 1 / 3), (0, 0.4), (0, 5 / 9), (0, 0.6), (0, 2 / 3)]
        actual = [(rule.label, rule.weight) for rule in classifier.rules]
        self.assertEqual(expected, actual)


def setup_multi_ruleset_classifier(rule_selection_method):
    ruleset_pos = utilities.create_numerical_test_ruleset_pos()
    ruleset_neg = utilities.create_numerical_test_ruleset_neg()
    classifier = RuleSetClassifier([ruleset_pos, ruleset_neg],
                                   rule_selection_method=rule_selection_method,
                                   confidence_metric=ConfidenceMetric.LAPLACE,
                                   weight_metric=WeightMetric.CONFIDENCE)
    X, y = utilities.create_numerical_test_data(20)
    classifier.update_rules_with_metrics(X, y)
    return classifier
