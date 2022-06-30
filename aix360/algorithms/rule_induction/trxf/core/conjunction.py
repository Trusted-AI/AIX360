from typing import Dict, Any, List

from aix360.algorithms.rule_induction.trxf.core.boolean_evaluator import BooleanEvaluator
from aix360.algorithms.rule_induction.trxf.core.predicate import Predicate
from aix360.algorithms.rule_induction.trxf.core.utils import set_equality


class Conjunction(BooleanEvaluator):
    def __init__(self, predicate_list: List[Predicate]):
        """
        @param predicate_list: a list of predicates intrpreted as a conjunction
        """
        self._predicates = predicate_list

    @property
    def predicates(self):
        return self._predicates

    def add_predicate(self, pred: Predicate):
        self._predicates.append(pred)

    def delete_predicate(self, pred: Predicate):
        self._predicates.remove(pred)

    def evaluate(self, assignment: Dict[str, Any]) -> bool:
        """
        Evaluate the truth value of the conjunction w.r.t. the variable assignment

        @param assignment: dict mapping variable name to value
        @return: bool truth value of the predicate
        """       
        for pred in self.predicates:
            if not pred.evaluate(assignment):
                return False
        return True

    def __repr__(self):
        return '%s(%r)' %(self.__class__, self._predicates)

    def __str__(self):
        if len(self.predicates) == 0:
            return 'true'
        else:
            return ' ^ '.join(['['+str(x)+']' for x in self._predicates])

    def __eq__(self, other):
        return set_equality(self.predicates, other.predicates)

    def __len__(self):
        return len(self.predicates)
