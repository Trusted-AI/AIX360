from typing import Dict

import numpy as np
import pandas as pd

from aix360.algorithms.rule_induction.trxf.classifier import ruleset_classifier
from aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier import RuleSetClassifier
from aix360.algorithms.rule_induction.trxf.core import Conjunction, Relation
from aix360.algorithms.rule_induction.trxf.pmml_export import models
from aix360.algorithms.rule_induction.trxf.pmml_export.models.data_dictionary import Value
from aix360.algorithms.rule_induction.trxf.pmml_export.reader import AbstractReader
from aix360.algorithms.rule_induction.trxf.pmml_export.models import SimplePredicate, Operator, CompoundPredicate, \
    BooleanOperator


class TrxfReader(AbstractReader):
    def __init__(self, data_dictionary=None):
        self._data_dictionary = data_dictionary

    @property
    def data_dictionary(self):
        return self._data_dictionary

    def read(self, trxf_classifier: RuleSetClassifier) -> models.SimplePMMLRuleSetModel:
        """
        Translate a TRXF RuleSetClassifier to an internal SimplePMMLRuleSetModel
        """
        trxf_to_pmml_rule_selection_map = {
            ruleset_classifier.RuleSelectionMethod.FIRST_HIT: models.RuleSelectionMethod.firstHit,
            ruleset_classifier.RuleSelectionMethod.WEIGHTED_SUM: models.RuleSelectionMethod.weightedSum,
            ruleset_classifier.RuleSelectionMethod.WEIGHTED_MAX: models.RuleSelectionMethod.weightedMax
        }
        trxf_rules = trxf_classifier.rules
        mining_schema = _extract_mining_schema(trxf_rules)
        simple_rules = _convert_to_simple_rules(trxf_rules)
        ruleset = models.RuleSet(
            ruleSelectionMethod=[trxf_to_pmml_rule_selection_map[trxf_classifier.rule_selection_method]],
            rules=simple_rules,
            defaultScore=str(trxf_classifier.default_label))
        rule_set_model = models.RuleSetModel(miningSchema=mining_schema, ruleSet=ruleset)
        assert self._data_dictionary is not None
        return models.SimplePMMLRuleSetModel(dataDictionary=self._data_dictionary, ruleSetModel=rule_set_model)

    def load_data_dictionary(self, X: pd.DataFrame, values: Dict = None):
        """
        Extract the data dictionary from a feature dataframe, and store it

        @param X: Input dataframe
        @param values: A dict mapping column name to a list of possible categorical values. It will be inferred from X if not provided.
        """
        dtypes = X.dtypes
        data_fields = []
        for index, value in dtypes.items():
            vals = None
            if np.issubdtype(value, np.integer):
                data_type = models.DataType.integer
                op_type = models.OpType.ordinal
            elif np.issubdtype(value, np.double):
                data_type = models.DataType.double
                op_type = models.OpType.continuous
            elif np.issubdtype(value, np.floating):
                data_type = models.DataType.float
                op_type = models.OpType.continuous
            elif np.issubdtype(value, np.bool_):
                data_type = models.DataType.boolean
                op_type = models.OpType.categorical
            else:
                data_type = models.DataType.string
                op_type = models.OpType.categorical
                vals = values[index] if values is not None and index in values else list(X[index].unique())
            wrapped_vals = list(map(lambda v: Value(v), vals)) if vals is not None else vals
            data_fields.append(models.DataField(name=str(index), optype=op_type, dataType=data_type, values=wrapped_vals))
        self._data_dictionary = models.DataDictionary(data_fields)


def _convert_to_simple_rules(trxf_rules):
    simple_rules = []
    for rule in trxf_rules:
        predicate = _convert_to_pmml_predicate(rule.conjunction)
        confidence = rule.confidence if rule.confidence is not None else models.DEFAULT_CONFIDENCE
        weight = rule.weight if rule.weight is not None else models.DEFAULT_WEIGHT
        simple_rule = models.SimpleRule(predicate=predicate, score=str(rule.label), id=str(rule.conjunction),
                                        recordCount=rule.record_count,
                                        nbCorrect=rule.nb_correct, confidence=confidence, weight=weight)
        simple_rules.append(simple_rule)
    return simple_rules


def _convert_to_pmml_predicate(trxf_conjunction: Conjunction):
    trxf_to_pmml_op = {
        Relation.EQ: Operator.equal,
        Relation.NEQ: Operator.notEqual,
        Relation.LT: Operator.lessThan,
        Relation.LE: Operator.lessOrEqual,
        Relation.GT: Operator.greaterThan,
        Relation.GE: Operator.greaterOrEqual
    }
    simple_predicates = [SimplePredicate(operator=trxf_to_pmml_op[trxf_predicate.relation],
                                         value=str(trxf_predicate.value),
                                         field=str(trxf_predicate.feature.variable_names[0]))
                         for trxf_predicate in trxf_conjunction.predicates]
    return CompoundPredicate(simplePredicates=simple_predicates, booleanOperator=BooleanOperator.and_)


def _extract_mining_schema(trxf_rules):
    mining_fields = {}
    for rule in trxf_rules:
        for predicate in rule.conjunction.predicates:
            name = predicate.feature.variable_names[0]
            if name not in mining_fields:
                mining_field = models.MiningField(name=name)
                mining_fields[name] = mining_field
    return models.MiningSchema(miningFields=list(mining_fields.values()))
