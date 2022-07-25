from unittest import TestCase
from aix360.algorithms.rule_induction.trxf.core.feature import Feature
import aix360.algorithms.rule_induction.trxf.core.predicate as predicate
import aix360.algorithms.rule_induction.trxf.core.conjunction as conjunction
import aix360.algorithms.rule_induction.trxf.core.dnf_ruleset as ruleset


class TestDnfRuleSet(TestCase):

    def reset_values(self):

        self.feature1 = Feature('x1')
        self.feature2 = Feature('x2')
        self.feature3 = Feature('x3')
        self.feature4 = Feature('x4')
        self.feature5 = Feature('x5')
        self.feature6 = Feature('x6')

        self.pred1 = predicate.Predicate(self.feature1, predicate.Relation.LT, 25)
        self.pred2 = predicate.Predicate(self.feature2, predicate.Relation.GT, 50000)
        self.pred3 = predicate.Predicate(self.feature3, predicate.Relation.EQ, 'DUMMY')
        self.pred4 = predicate.Predicate(self.feature4, predicate.Relation.NEQ, 10.25)
        self.pred5 = predicate.Predicate(self.feature5, predicate.Relation.EQ, False)
        self.pred6 = predicate.Predicate(self.feature6, predicate.Relation.GE, -1)

        # Add conjunctions
        self.conjunction1 = conjunction.Conjunction([self.pred1, self.pred2, self.pred3])
        self.conjunction2 = conjunction.Conjunction([])
        self.conjunction3 = conjunction.Conjunction([self.pred1, self.pred2, self.pred3])   # Same as conjunction1
        self.conjunction4 = conjunction.Conjunction([self.pred3, self.pred5, self.pred6])
        self.conjunction5 = conjunction.Conjunction([self.pred4])
        self.conjunction6 = conjunction.Conjunction([self.pred5, self.pred6])

    def test_conjunctions(self):
        self.reset_values()
        self.assertListEqual(ruleset.DnfRuleSet([self.conjunction1, self.conjunction4], 'ThenPart').list_conjunctions(),
                             [self.conjunction1, self.conjunction4])
        self.assertListEqual(ruleset.DnfRuleSet([self.conjunction1, self.conjunction3], 'ThenPart').list_conjunctions(),
                             [self.conjunction1])
        self.assertListEqual(ruleset.DnfRuleSet([self.conjunction2], 'ThenPart').list_conjunctions(),
                             [self.conjunction2])

    def test_ruleset_modifications(self):

        self.reset_values()

        # Adding to empty ruleset
        rs = ruleset.DnfRuleSet([], 'ThenPart')
        rs.add_conjunction(self.conjunction1)
        self.assertListEqual(rs.list_conjunctions(), [self.conjunction1])

        # Adding a duplicate conjunction
        rs = ruleset.DnfRuleSet([self.conjunction1], 'ThenPart')
        rs.add_conjunction(self.conjunction3)
        self.assertListEqual(rs.list_conjunctions(), [self.conjunction1])

        # Removing a conjunction
        rs = ruleset.DnfRuleSet([self.conjunction1, self.conjunction4], 'ThenPart')
        rs.remove_conjunction(self.conjunction1)
        self.assertListEqual(rs.list_conjunctions(), [self.conjunction4])

        # Removing a conjunction that does not exist
        rs = ruleset.DnfRuleSet([self.conjunction1, self.conjunction4], 'ThenPart')
        self.assertRaises(ValueError, rs.remove_conjunction, self.conjunction2)

    def test_evaluate(self):

        self.reset_values()

        # One rule satisfied and one not satisfied
        values = {'x1': 25, 'x2': 70000, 'x3': 'DUMMY', 'x4': 10.24, 'x5': False, 'x6': -1}
        rs = ruleset.DnfRuleSet([self.conjunction1, self.conjunction4], 'ThenPart')
        self.assertEqual(rs.evaluate(values), True)

        # Empty ruleset
        rs = ruleset.DnfRuleSet([], 'ThenPart')
        self.assertEqual(rs.evaluate(values), False)

        # Ruleset containing an empty rule (all other rules are violated)
        values = {'x1': 25, 'x2': 70000, 'x3': 'DUMMY', 'x4': 10.25, 'x5': True, 'x6': -2}
        rs = ruleset.DnfRuleSet([self.conjunction2, self.conjunction5, self.conjunction6], 'ThenPart')
        self.assertEqual(rs.evaluate(values), True)

        # No rule satsified
        values = {'x1': 25, 'x2': 70000, 'x3': 'DUMMY', 'x4': 10.25, 'x5': True, 'x6': -1}
        rs = ruleset.DnfRuleSet([self.conjunction5, self.conjunction6], 'ThenPart')
        self.assertEqual(rs.evaluate(values), False)
