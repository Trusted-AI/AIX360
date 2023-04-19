from typing import Dict

import pandas as pd

from aix360.algorithms.rule_induction.trxf.classifier import ruleset_classifier
from aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier import RuleSetClassifier
from aix360.algorithms.rule_induction.trxf.pmml_export import models
from aix360.algorithms.rule_induction.trxf.pmml_export.reader import AbstractReader
from aix360.algorithms.rule_induction.trxf.pmml_export.utilities import extract_data_dictionary, trxf_to_pmml_predicate


class TrxfRuleSetReader(AbstractReader):
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
        self._data_dictionary = extract_data_dictionary(X, values)


def _convert_to_simple_rules(trxf_rules):
    simple_rules = []
    for rule in trxf_rules:
        predicate = trxf_to_pmml_predicate(rule.conjunction)
        confidence = rule.confidence if rule.confidence is not None else models.DEFAULT_CONFIDENCE
        weight = rule.weight if rule.weight is not None else models.DEFAULT_WEIGHT
        simple_rule = models.SimpleRule(predicate=predicate, score=str(rule.label), id=str(rule.conjunction),
                                        recordCount=rule.record_count,
                                        nbCorrect=rule.nb_correct, confidence=confidence, weight=weight)
        simple_rules.append(simple_rule)
    return simple_rules


def _extract_mining_schema(trxf_rules):
    mining_fields = {}
    for rule in trxf_rules:
        for predicate in rule.conjunction.predicates:
            name = predicate.feature.variable_names[0]
            if name not in mining_fields:
                mining_field = models.MiningField(name=name)
                mining_fields[name] = mining_field
    return models.MiningSchema(miningFields=list(mining_fields.values()))
