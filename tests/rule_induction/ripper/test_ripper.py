import logging
import sys
import unittest

import pandas as pd

from aix360.algorithms.rule_induction.ripper.ripper import RipperExplainer
from aix360.algorithms.rule_induction.trxf.core.conjunction import Conjunction
from aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from aix360.algorithms.rule_induction.trxf.core.feature import Feature
from aix360.algorithms.rule_induction.trxf.core.predicate import Predicate, Relation


def create_test_df():
    df = pd.DataFrame({
        'int_col': [1, 2],
        'float_col': [1.2, 4.5],
        'str_col': ['foo', 'bar'],
        'target': ['True', 'False']
    })
    df.astype({'int_col': 'int64', 'float_col': 'float64', 'str_col': 'str', 'target': 'str'})
    return df


def create_degenerate_test_df():
    df = pd.DataFrame({
        'int_col': [1, 1],
        'float_col': [1.2, 1.2],
        'str_col': ['foo', 'foo'],
        'target': ['True', 'False']
    })
    df.astype({'int_col': 'int64', 'float_col': 'float64', 'str_col': 'str', 'target': 'str'})
    return df


def create_degenerate_test_df2():
    df = pd.DataFrame({
        'int_col': [1, 1, 1, 1, 1, 1, 1],
        'float_col': [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
        'str_col': ['foo', 'foo', 'foo', 'foo', 'foo', 'foo', 'foo'],
        'target': ['True', 'False', 'False', 'False', 'False', 'False', 'False']
    })
    df.astype({'int_col': 'int64', 'float_col': 'float64', 'str_col': 'str', 'target': 'str'})
    return df


class TestRipper(unittest.TestCase):

    def test_fit_pos_value_is_respected(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        df = create_test_df()
        TARGET_LABEL = 'target'
        POS_VALUE = 'True'
        x_train = df.drop(columns=[TARGET_LABEL])
        y_train = df[TARGET_LABEL]

        estimator = RipperExplainer()
        estimator.fit(x_train, y_train, target_label=POS_VALUE)
        ruleset = estimator.explain()
        actual_pos_value = ruleset.then_part

        self.assertEqual(actual_pos_value, POS_VALUE)

    def test_trxf_export(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logger = logging.getLogger(__name__)

        df = create_test_df()
        TARGET_LABEL = 'target'
        POS_VALUE = 'True'
        x_train = df.drop(columns=[TARGET_LABEL])
        y_train = df[TARGET_LABEL]

        estimator = RipperExplainer()
        estimator.fit(x_train, y_train, target_label=POS_VALUE)
        actual_rule_set = estimator.explain()
        logger.info(actual_rule_set)

        feature = Feature('int_col')
        predicate = Predicate(feature, Relation.EQ, 1)
        conjunction = Conjunction([predicate])
        expected_rule_set = DnfRuleSet([conjunction], 'True')
        self.assertEqual(actual_rule_set, expected_rule_set)

        assignment = {'int_col': 1, 'float_col': 1.2, 'str_col': 'foo'}
        result = actual_rule_set.evaluate(assignment)
        self.assertTrue(result)

    def test_always_true_rule(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logger = logging.getLogger(__name__)

        df = create_degenerate_test_df()
        TARGET_LABEL = 'target'
        POS_VALUE = 'True'
        x_train = df.drop(columns=[TARGET_LABEL])
        y_train = df[TARGET_LABEL]

        estimator = RipperExplainer()
        estimator.fit(x_train, y_train, target_label=POS_VALUE)

        actual_rule_set = estimator.explain()
        logger.info(actual_rule_set)

        expected_conjunction = Conjunction([])
        expected_rule_set = DnfRuleSet([expected_conjunction], 'True')
        self.assertEqual(actual_rule_set, expected_rule_set)

        assignment = {'int_col': 1, 'float_col': 1.2, 'str_col': 'foo'}
        result = actual_rule_set.evaluate(assignment)
        self.assertTrue(result)

    def test_always_false_rule(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logger = logging.getLogger(__name__)

        df = create_degenerate_test_df2()
        TARGET_LABEL = 'target'
        POS_VALUE = 'True'
        x_train = df.drop(columns=[TARGET_LABEL])
        y_train = df[TARGET_LABEL]

        estimator = RipperExplainer()
        estimator.fit(x_train, y_train, target_label=POS_VALUE)

        actual_rule_set = estimator.explain()
        logger.info(actual_rule_set)

        expected_rule_set = DnfRuleSet([], 'True')
        self.assertEqual(actual_rule_set, expected_rule_set)

        assignment = {'int_col': 1, 'float_col': 1.2, 'str_col': 'foo'}
        result = actual_rule_set.evaluate(assignment)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
