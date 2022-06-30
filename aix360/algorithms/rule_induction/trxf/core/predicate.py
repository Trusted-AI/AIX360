import math
from enum import Enum
from typing import Dict, Any

from aix360.algorithms.rule_induction.trxf.core.boolean_evaluator import BooleanEvaluator
from aix360.algorithms.rule_induction.trxf.core.feature import Feature, is_number

ABS_TOL = 1e-5


class Relation(Enum):
    NEQ = '!='
    EQ = '=='
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='

    def __neg__(self):
        neg_table = {
            self.NEQ: self.EQ,
            self.EQ: self.NEQ,
            self.LT: self.GE,
            self.GE: self.LT,
            self.LE: self.GT,
            self.GT: self.LE
        }
        return neg_table[self]


class Predicate(BooleanEvaluator):
    def __init__(self, feature: Feature, relation: Relation, value: Any):
        """
        @param feature: the LHS of the relation
        @param value: the RHS of the relation
        """
        self._feature = feature
        self._relation = relation
        self._value = value
        _validate_predicate(self._relation, self._value)

    @property
    def feature(self):
        return self._feature

    @property
    def relation(self):
        return self._relation

    @property
    def value(self):
        return self._value

    def evaluate(self, assignment: Dict[str, Any]) -> bool:
        """
        Evaluate the truth value of the predicate w.r.t. the variable assignment

        @param assignment: dict mapping variable name to value
        @return: bool truth value of the predicate
        """
        feature_val = self.feature.evaluate(assignment)
        if self.relation == Relation.EQ:
            return math.isclose(feature_val, self.value, abs_tol=ABS_TOL) if is_number(self.value) \
                else feature_val == self.value
        elif self.relation == Relation.NEQ:
            return feature_val != self.value
        elif self.relation == Relation.GE:
            return feature_val >= self.value
        elif self.relation == Relation.GT:
            return feature_val > self.value
        elif self.relation == Relation.LE:
            return feature_val <= self.value
        elif self.relation == Relation.LT:
            return feature_val < self.value
        else:
            raise ValueError('Unknown relation operator {}'.format(self.relation))

    def __neg__(self):
        if self.value is True or self.value is False:
            return Predicate(self.feature, self.relation, not self.value)
        return Predicate(self.feature, -self.relation, self.value)

    def __repr__(self):
        return '%s(%r, %r, %r)' % (self.__class__, self.feature, self.relation, self.value)

    def __str__(self):
        return ' '.join([str(self.feature), str(self.relation.value), str(self.value)])

    def __eq__(self, other):
        return self.feature == other.feature and self.relation == other.relation and self.value == other.value


def _validate_predicate(relation, value):
    if (isinstance(value, str) or isinstance(value, bool)) and relation != Relation.EQ and relation != Relation.NEQ:
        raise ValueError('String or boolean value "{}" can only be compared through "==" or "!=" but was "{}"'
                         .format(value, relation.value))


