from typing import Dict, List, Any

from aix360.algorithms.rule_induction.trxf.core.boolean_evaluator import BooleanEvaluator
from aix360.algorithms.rule_induction.trxf.core.conjunction import Conjunction
from aix360.algorithms.rule_induction.trxf.core.utils import set_equality


class DnfRuleSet(BooleanEvaluator):
    """
    This class implements a ruleset in the disjunctive normal form: c_1 v c_2 v .... v c_k. Here, each c_i is a
    conjunction of predicates, i.e., c_i = p_{i1} & p_{i2} & ... & p_{i n_i}. All the conjunctions share the same
    conclusion, specified by the then_part of the ruleset.
    """

    def __init__(self, conjunctions: List[Conjunction], then_part: Any) -> None:
        """
        @param conjunctions: A list containing conjunctions in this ruleset
        @param then_part: The then part of this RuleSet.
        """
        self._then_part = then_part
        self._conjunctions = []
        for conjunction in conjunctions:
            if conjunction not in self._conjunctions:
                self._conjunctions.append(conjunction)

    @property
    def conjunctions(self):
        return self._conjunctions

    @property
    def then_part(self):
        return self._then_part

    def add_conjunction(self, conjunction: Conjunction) -> None:
        if conjunction not in self._conjunctions:
            self._conjunctions.append(conjunction)

    def remove_conjunction(self, conjunction: Conjunction) -> None:
        self._conjunctions.remove(conjunction)

    def list_conjunctions(self) -> List[Conjunction]:
        return self.conjunctions

    def evaluate(self, assignment: Dict[str, Any]) -> bool:
        for conjunction in self.conjunctions:
            if conjunction.evaluate(assignment):
                return True
        return False

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__, self._conjunctions, self._then_part)

    def __str__(self):
        if len(self.conjunctions) == 0:
            disjunction = 'false'
        else:
            disjunction = ' v\n'.join(['(' + str(x) + ')' for x in self._conjunctions])
        return 'if\n' + disjunction + '\nthen\n' + str(self._then_part)

    def __eq__(self, other):
        return set_equality(self.conjunctions, other.conjunctions) and self.then_part == other.then_part

    def __len__(self):
        return len(self.conjunctions)
