from unittest import TestCase

import numpy as np
import pandas as pd

from aix360.algorithms.rule_induction.ripper.ripper_ruleset_generator import RipperRuleSetGenerator
from aix360.algorithms.rule_induction.trxf.core.conjunction import Conjunction
from aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from aix360.algorithms.rule_induction.trxf.core.feature import Feature
import aix360.algorithms.rule_induction.trxf.core.predicate as predicate


def create_test_data(n_rows):
    np.random.seed(1)
    X = pd.DataFrame(np.random.randn(n_rows, 2), columns=['x1', 'x2'])
    y = pd.Series(np.random.binomial(1, 0.4, n_rows))
    return X, y


class TestRipperRuleSetGenerator(TestCase):
    def test_generate(self):
        X, y = create_test_data(10)
        generator = RipperRuleSetGenerator()
        p1 = predicate.Predicate(Feature('x1'), predicate.Relation.LE, 0.31903909605709857)
        p2 = predicate.Predicate(Feature('x2'), predicate.Relation.LE, -0.8778584179213718)
        p3 = predicate.Predicate(Feature('x2'), predicate.Relation.GE, -0.2493703754774101)
        p4 = predicate.Predicate(Feature('x1'), predicate.Relation.GE, 0.31903909605709857)
        c1 = Conjunction([p1, p2])
        c2 = Conjunction([p3, p4])
        expected = DnfRuleSet([c1, c2], 0)
        actual = generator.generate(X, y, 0, random_state=1)
        self.assertEqual(actual, expected)
