"""Actual TRXF components and imports; no comparison/prediction mocks."""
import os
from pathlib import Path
import sys
from unittest import TestCase, main

source_root = os.environ.get('AIX360_SOURCE_ROOT')
if source_root is None:
    target = Path('aix360/algorithms/rule_induction/trxf/classifier/ruleset_classifier.py')
    source_root = next((parent for parent in Path(__file__).resolve().parents
                        if (parent / target).is_file()), None)
    if source_root is None:
        raise FileNotFoundError('Set AIX360_SOURCE_ROOT to the source tree')
sys.path.insert(0, str(Path(source_root).resolve()))

from aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier import (
    RuleSetClassifier, RuleSelectionMethod,
)
from aix360.algorithms.rule_induction.trxf.core import Conjunction, DnfRuleSet, Feature, Predicate, Relation


def make_ruleset(threshold=0, label='positive'):
    return DnfRuleSet([Conjunction([Predicate(Feature('x'), Relation.GT, threshold)])], label)


def make_classifier(rule_sets=None, default_label='default'):
    if rule_sets is None:
        rule_sets = [make_ruleset()]
    return RuleSetClassifier(rule_sets, RuleSelectionMethod.FIRST_HIT, default_label=default_label)


class TestClassifierEquality(TestCase):
    def test_reflexive_equality(self):
        classifier = make_classifier()
        self.assertTrue(classifier == classifier)

    def test_separately_constructed_equal_components_compare_equal(self):
        left = make_classifier()
        right = make_classifier()
        self.assertIsNot(left, right)
        self.assertTrue(left == right)
        self.assertTrue(right == left)

    def test_different_predicates_compare_unequal(self):
        left = make_classifier([make_ruleset(threshold=0)])
        right = make_classifier([make_ruleset(threshold=2)])
        self.assertFalse(left == right)

    def test_different_rule_conclusions_compare_unequal(self):
        left = make_classifier([make_ruleset(label='positive')])
        right = make_classifier([make_ruleset(label='negative')])
        self.assertFalse(left == right)

    def test_different_defaults_compare_unequal(self):
        left = make_classifier(default_label='positive')
        right = make_classifier(default_label='negative')
        self.assertFalse(left == right)

    def test_outer_ruleset_list_order_is_preserved(self):
        left = make_classifier([make_ruleset(label='first'), make_ruleset(label='second')])
        right = make_classifier([make_ruleset(label='second'), make_ruleset(label='first')])
        self.assertFalse(left == right)
        # Both rules fire; FIRST_HIT demonstrates why list order is meaningful.
        self.assertEqual(left.predict({'x': 1}), 'first')
        self.assertEqual(right.predict({'x': 1}), 'second')

    def test_empty_rule_lists_and_none_defaults(self):
        left = make_classifier([], None)
        right = make_classifier([], None)
        self.assertTrue(left == right)
        self.assertFalse(left == make_classifier([], 'fallback'))
        self.assertFalse(left == make_classifier())

    def test_existing_core_comparison_and_first_hit_control(self):
        self.assertEqual(make_ruleset(), make_ruleset())
        self.assertNotEqual(make_ruleset(label='positive'), make_ruleset(label='negative'))
        classifier = make_classifier()
        self.assertEqual(classifier.predict({'x': 1}), 'positive')
        self.assertEqual(classifier.predict({'x': -1}), 'default')


if __name__ == '__main__':
    main()
