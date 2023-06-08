import unittest

import numpy as np

from aix360.algorithms.imd.rule import Rule
from aix360.algorithms.imd.utils import load_bc_dataset


class TestIMDExplainer(unittest.TestCase):

    def test_rule_class(self):
        rule1 = Rule(0, [('x1', '<=', 6), ('x2', '<=', 100), ('x1', '>', 1)], 0)
        rule2 = Rule(1, [('x1', '<=', 9), ('x2', '>', 100), ('x1', '>', 3)], 1)
        rule3 = Rule(2, [('x1', '<=', 2), ('x2', '<=', 90)], 1)

        rule31 = Rule(31, [('x2', '<=', 90), ('x1', '>', 1), ('x1', '<=', 2)], 1)
        rule2_ = Rule(100, [('x1', '>', 3), ('x2', '>', 100), ('x1', '<=', 9)], -1)

        self.assertTrue(rule31.check_equal_preds(rule1.intersection(rule3)), "check rule1 intersection rule3 predicate equivalence")
        self.assertTrue(rule2.check_equal_preds(rule2_), "check rule2 and rule2_ predicate equivalence")

        self.assertEqual(rule1.as_dict(),
                         {'x1': [1, 6], 'x2': [np.NINF, 100]}
                         )
        self.assertEqual(rule2.as_dict(),
                         {'x1': [3, 9], 'x2': [100, np.inf]}
                         )
        self.assertEqual(rule3.as_dict(),
                         {'x1': [np.NINF, 2], 'x2': [np.NINF, 90]}
                         )
        self.assertEqual(rule3.intersection(rule1).as_dict(),
                         {'x1': [1, 2], 'x2': [np.NINF, 90]}
                         )
        self.assertEqual(rule1.intersection(rule2).as_dict(), {})

    def test_imd(self):
        pass












if __name__=='__main__':
    unittest.main()