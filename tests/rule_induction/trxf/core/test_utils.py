from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from aix360.algorithms.rule_induction.trxf.core.utils import set_equality, batch_evaluate


class TestRuleSet(TestCase):

    def test_set_equality(self):
        # Empty lists
        self.assertEqual(set_equality([], []), True)
        self.assertEqual(set_equality([], [1, 2]), False)
        self.assertEqual(set_equality(['a', 1], []), False)

        # List with repeated elements
        self.assertEqual(set_equality([1, 2, 1, -1], [1, 2, -1]), True)
        self.assertEqual(set_equality([1, 2, 1, -1], [1, 2, -1, -1]), True)

        # Lists with different orders
        self.assertEqual(set_equality([1, 2, 3], [3, 1, 2]), True)

        # Unequal lists
        self.assertEqual(set_equality([1, 2, 1], [1, 2, -1]), False)
        self.assertEqual(set_equality([1, 2, 1, -1], [1, -1, -1]), False)

    def test_batch_evaluate(self):
        d = {'x1': [1, 2], 'x2': [-2, -1]}
        df = pd.DataFrame(data=d)

        def side_effect_fn(assignment):
            return True if assignment == {'x1': 1, 'x2': -2} else False

        mock_ruleset = DnfRuleSet([], 'class1')
        mock_ruleset.evaluate = MagicMock(side_effect=side_effect_fn)
        actual = batch_evaluate(mock_ruleset, df)
        expected = pd.Series([True, False])
        pd.testing.assert_series_equal(actual, expected)
