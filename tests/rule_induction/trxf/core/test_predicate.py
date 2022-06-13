from unittest import TestCase
from unittest.mock import MagicMock

import aix360.algorithms.rule_induction.trxf.core.predicate as predicate
from aix360.algorithms.rule_induction.trxf.core.feature import Feature


class TestPredicate(TestCase):
    def test_construct_categorical_predicate_with_inequality_should_raise(self):
        mock_feature = Feature('feature')
        self.assertRaises(ValueError, predicate.Predicate, mock_feature, predicate.Relation.GT, 'bar')
        self.assertRaises(ValueError, predicate.Predicate, mock_feature, predicate.Relation.GT, False)

    def test_evaluate_EQ_categorical(self):
        mock_feature = Feature('categorical_feature')
        mock_feature.evaluate = MagicMock(return_value='foo')
        assignment = {'categorical_feature': 'foo'}

        isFoo = predicate.Predicate(mock_feature, predicate.Relation.EQ, 'foo')
        self.assertTrue(isFoo.evaluate(assignment))
        mock_feature.evaluate.assert_called_with(assignment)

        isBar = predicate.Predicate(mock_feature, predicate.Relation.EQ, 'bar')
        self.assertFalse(isBar.evaluate(assignment))
        mock_feature.evaluate.assert_called_with(assignment)

    def test_evaluate_EQ_numerical(self):
        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=3.2)
        assignment = {'feature': 3.2}

        pred = predicate.Predicate(mock_feature, predicate.Relation.EQ, 3.2 + 1e-6)
        self.assertTrue(pred.evaluate(assignment))
        mock_feature.evaluate.assert_called_with(assignment)

        pred = predicate.Predicate(mock_feature, predicate.Relation.EQ, 3.2 + 1e-5)
        self.assertFalse(pred.evaluate(assignment))
        mock_feature.evaluate.assert_called_with(assignment)

    def test_evaluate_LT(self):
        mock_feature = Feature('numerical_feature')
        mock_feature.evaluate = MagicMock(return_value=3.4)
        assignment = {'numerical_feature': 3.4}

        isLessThan3 = predicate.Predicate(mock_feature, predicate.Relation.LT, 3)
        self.assertFalse(isLessThan3.evaluate(assignment))

        isLessThan4 = predicate.Predicate(mock_feature, predicate.Relation.LT, 4)
        self.assertTrue(isLessThan4.evaluate(assignment))

        self.assertRaises(ValueError, predicate.Predicate(mock_feature, 'LT', 4).evaluate, assignment)

    def test_negate(self):
        mock_feature = Feature('categorical_feature')

        isFoo = predicate.Predicate(mock_feature, predicate.Relation.EQ, 'foo')
        isNotFoo = predicate.Predicate(mock_feature, predicate.Relation.NEQ, 'foo')
        self.assertEqual(-isFoo, isNotFoo)

    def test_negate_bool(self):
        mock_feature = Feature('categorical_feature')

        isFoo = predicate.Predicate(mock_feature, predicate.Relation.EQ, True)
        isNotFoo = predicate.Predicate(mock_feature, predicate.Relation.EQ, False)
        self.assertEqual(-isFoo, isNotFoo)
