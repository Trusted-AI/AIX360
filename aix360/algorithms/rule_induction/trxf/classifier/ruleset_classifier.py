import enum
from collections import namedtuple
from typing import Dict, List, Any

import pandas as pd

from aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from aix360.algorithms.rule_induction.trxf.metrics import compute_rule_metrics

Rule = namedtuple('Rule', ['conjunction', 'label', 'record_count', 'nb_correct', 'confidence', 'weight'])


class RuleSelectionMethod(enum.Enum):
    FIRST_HIT = enum.auto()
    WEIGHTED_MAX = enum.auto()
    WEIGHTED_SUM = enum.auto()


class ConfidenceMetric(enum.Enum):
    LAPLACE = enum.auto()
    ACCURACY = enum.auto()
    CONSTANT = enum.auto()


class WeightMetric(enum.Enum):
    CONFIDENCE = enum.auto()
    CONSTANT = enum.auto()


class RuleSetClassifier:
    """
    This class implements a simple classifier constructed from a list of DNF rule sets.
    Based on the specified RuleSelectionMethod, this classifier predicts the label for a given input.
    """

    def __init__(self,
                 rule_sets: List[DnfRuleSet],
                 rule_selection_method: RuleSelectionMethod,
                 confidence_metric: ConfidenceMetric = None,
                 weight_metric: WeightMetric = None,
                 default_label: Any = None):
        """
        @param rule_sets: A list of dnf rule sets
        @param rule_selection_method: Conflict resolution strategy for multiple fired rules
        @param confidence_metric: Rule confidence definition to use
        @param weight_metric: Rule weight definition to use depending on RuleSelectionMethod
        @param default_label: Optional default label to be returned when no rule set matches the data input.
        """
        self._rule_sets = rule_sets
        self._rule_selection_method = rule_selection_method
        self._default_label = default_label
        self._confidence_metric = confidence_metric
        self._weight_metric = weight_metric
        self._rules = [Rule(conjunction=c,
                            label=ruleset.then_part,
                            record_count=None,
                            nb_correct=None,
                            confidence=None,
                            weight=None)
                       for ruleset in self._rule_sets for c in ruleset.conjunctions]

    @property
    def rules(self):
        return self._rules

    @property
    def rule_selection_method(self):
        return self._rule_selection_method

    @property
    def default_label(self):
        return self._default_label

    def update_rules_with_metrics(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Compute rule level metrics for the provided input, and update the Rules with them
        @param X: pandas data frame to compute required confidence score for PMML exported rules
        @param y: pandas series to hold labels to compute required confidence score for PMML exported rules
        """
        new_rules = []
        for rule_set in self._rule_sets:
            metric_list = compute_rule_metrics(rule_set, X, y)
            for rule in metric_list:
                confidence = self._compute_confidence(rule)
                weight = self._compute_weight(rule)
                new_rules.append(
                    Rule(conjunction=rule.conjunction, label=rule_set.then_part, record_count=rule.n_records,
                         nb_correct=rule.n_correct, confidence=confidence, weight=weight))
        self._rules = new_rules

    def predict(self, assignment: Dict[str, Any]) -> Any:
        """
        Predict the label for the given assignment
        @param assignment: A valuation of the variables that occur in the rule sets
        """
        if self.rule_selection_method == RuleSelectionMethod.FIRST_HIT:
            for rule in self.rules:
                if rule.conjunction.evaluate(assignment):
                    return rule.label
            return self.default_label
        elif self.rule_selection_method == RuleSelectionMethod.WEIGHTED_SUM:
            assert self.rules[0].weight is not None
            label_weights = {}
            for rule in self.rules:
                if rule.label not in label_weights:
                    label_weights[rule.label] = rule.weight
                else:
                    label_weights[rule.label] += rule.weight
            return max(label_weights, key=label_weights.get)
        elif self.rule_selection_method == RuleSelectionMethod.WEIGHTED_MAX:
            assert self.rules[0].weight is not None
            label_weights = {}
            for rule in self.rules:
                if rule.label not in label_weights:
                    label_weights[rule.label] = rule.weight
                else:
                    label_weights[rule.label] = max(rule.weight, label_weights[rule.label])
            return max(label_weights, key=label_weights.get)
        else:
            raise ValueError("Unknown rule selection method: {}".format(self.rule_selection_method))

    def _compute_confidence(self, rule):
        if self._confidence_metric is None:
            confidence = None
        elif self._confidence_metric == ConfidenceMetric.LAPLACE:
            confidence = rule.laplace_estimate
        elif self._confidence_metric == ConfidenceMetric.ACCURACY:
            confidence = rule.n_correct / rule.n_records
        elif self._confidence_metric == ConfidenceMetric.CONSTANT:
            confidence = 1
        else:
            raise ValueError("Unknown confidence metric: {}".format(self._confidence_metric))
        return confidence

    def _compute_weight(self, rule):
        if self._weight_metric == WeightMetric.CONFIDENCE:
            weight = self._compute_confidence(rule)
        elif self._weight_metric == WeightMetric.CONSTANT:
            weight = 1
        else:
            raise ValueError("Unknown weight metric: {}".format(self._weight_metric))
        return weight

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__, self._rule_sets, self._default_label)

    def __str__(self):
        body = '\n'.join([str(rule_set) for rule_set in self._rule_sets])
        else_clause = 'else ' + str(self.default_label)
        return '\n'.join([body, else_clause])

    def __eq__(self, other):
        return (self._rule_sets == other.rule_sets) and (self.default_label == other.default_label)
