from unittest import TestCase
import copy

from aix360.algorithms.rule_induction.trxf.core.feature import Feature
import aix360.algorithms.rule_induction.trxf.core.predicate as predicate
import aix360.algorithms.rule_induction.trxf.core.conjunction as conjunction


class TestConjunction(TestCase):
    def test_conjunction_ops(self):
        feature_1 = Feature('age')
        feature_2 = Feature('estimated_income')
        pred_1 = predicate.Predicate(feature_1, predicate.Relation.LT, 27)
        pred_2 = predicate.Predicate(feature_2, predicate.Relation.GE, 80000)
        conjunction_1 = conjunction.Conjunction([pred_1, pred_2])

        actual = len(conjunction_1)
        expected = 2
        self.assertEqual(actual, expected)

        actual = conjunction_1.predicates
        expected = [pred_1, pred_2]
        self.assertListEqual(actual, expected)

        old_conjunction_1 = copy.deepcopy(conjunction_1)

        conjunction_1.delete_predicate(pred_1)
        actual = conjunction_1.predicates
        expected = [pred_2]
        self.assertListEqual(actual, expected)

        conjunction_1.add_predicate(pred_1)
        actual = conjunction_1.predicates
        expected = [pred_2, pred_1]
        self.assertListEqual(actual, expected)
        self.assertTrue(conjunction_1 == old_conjunction_1)

    def test_evaluate(self):
        feature_1 = Feature('age')
        feature_2 = Feature('estimated_income')
        my_val = {'age': 25, 'estimated_income': 70000}
        pred_1 = predicate.Predicate(feature_1, predicate.Relation.LT, 27)
        pred_2 = predicate.Predicate(feature_2, predicate.Relation.GE, 80000)
        pred_3 = -pred_2
        conjunction_1 = conjunction.Conjunction([pred_1, pred_2])
        conjunction_2 = conjunction.Conjunction([pred_1, pred_3])

        self.assertFalse(conjunction_1.evaluate(my_val))
        self.assertTrue(conjunction_2.evaluate(my_val))
