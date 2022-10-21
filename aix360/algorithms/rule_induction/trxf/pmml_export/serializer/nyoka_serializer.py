import datetime
import io

import nyoka.PMML44 as nyoka_pmml  # noqa
import nyoka.base.constants as nyoka_constants

from .abstract_serializer import AbstractSerializer
from .. import __version__ as version
from .. import models as models


class NyokaSerializer(AbstractSerializer):
    APPLICATION_NAME = 'SimpleRuleSetExport'
    COPYRIGHT_STRING = 'Copyright IBM Corp'

    def __init__(self, timestamp: datetime = None):
        self._timestamp = timestamp

    def serialize(self, simple_pmml_ruleset_model: models.SimplePMMLRuleSetModel, timestamp: datetime = None) -> str:
        pmml_model = self._nyoka_pmml_model(simple_pmml_ruleset_model)
        string_io = io.StringIO()
        pmml_model.export(outfile=string_io, level=0)
        return string_io.getvalue()

    def serialize_scorecard(self, scorecard: models.Scorecard, timestamp: datetime = None) -> str:
        pmml_model = self._nyoka_scorecard(scorecard)
        string_io = io.StringIO()
        pmml_model.export(outfile=string_io, level=0)
        return string_io.getvalue()

    def _nyoka_scorecard(self, scorecard: models.Scorecard) -> nyoka_pmml.PMML:
        return nyoka_pmml.PMML(
            version=nyoka_constants.PMML_SCHEMA.VERSION,
            Header=nyoka_pmml.Header(
                copyright=NyokaSerializer.COPYRIGHT_STRING,
                description=nyoka_constants.HEADER_INFO.DEFAULT_DESCRIPTION,
                Timestamp=nyoka_pmml.Timestamp(datetime.datetime.now() if self._timestamp is None else self._timestamp),
                Application=nyoka_pmml.Application(
                    name=NyokaSerializer.APPLICATION_NAME, version=version.version)),
            DataDictionary=None if scorecard.dataDictionary is None else self._nyoka_data_dictionary(
                scorecard.dataDictionary),
            Scorecard=None if scorecard is None else [
                self._nyoka_scorecard_model(scorecard)])

    def _nyoka_scorecard_model(self, scorecard: models.Scorecard) -> nyoka_pmml.Scorecard:
        return nyoka_pmml.Scorecard(
            functionName='regression',
            algorithmName='ScoreCard',
            MiningSchema=None if scorecard.miningSchema is None else self._nyoka_mining_schema(
                scorecard.miningSchema),
            initialScore=scorecard.initialScore,
            useReasonCodes="false",
            Output=None if scorecard.output is None else nyoka_pmml.Output(
                OutputField=[
                    nyoka_pmml.OutputField(
                        name=outputField.name,
                        feature=outputField.feature,
                        dataType=outputField.dataType.name,
                        optype=outputField.optype.name) for outputField in scorecard.output.outputFields]),
            Characteristics=None if scorecard.characteristics is None else self._nyoka_pmml_characteristics(
                scorecard.characteristics))

    def _nyoka_pmml_model(self, simple_pmml_ruleset_model: models.SimplePMMLRuleSetModel) -> nyoka_pmml.PMML:
        timestamp = datetime.datetime.now() if self._timestamp is None else self._timestamp
        return nyoka_pmml.PMML(
            version=nyoka_constants.PMML_SCHEMA.VERSION,
            Header=nyoka_pmml.Header(
                copyright=NyokaSerializer.COPYRIGHT_STRING,
                description=nyoka_constants.HEADER_INFO.DEFAULT_DESCRIPTION,
                Timestamp=nyoka_pmml.Timestamp(timestamp),
                Application=nyoka_pmml.Application(
                    name=NyokaSerializer.APPLICATION_NAME, version=version.version)),
            DataDictionary=None if simple_pmml_ruleset_model.dataDictionary is None else self._nyoka_data_dictionary(
                simple_pmml_ruleset_model.dataDictionary),
            RuleSetModel=None if simple_pmml_ruleset_model.ruleSetModel is None else [
                self._nyoka_rule_set_model(simple_pmml_ruleset_model.ruleSetModel)])

    def _nyoka_data_dictionary(self, data_dictionary: models.DataDictionary) -> nyoka_pmml.DataDictionary:
        return nyoka_pmml.DataDictionary(
            numberOfFields=0 if data_dictionary.dataFields is None else len(data_dictionary.dataFields),
            DataField=None if data_dictionary.dataFields is None else [
                nyoka_pmml.DataField(name=f.name, optype=f.optype.name, dataType=f.dataType.name)
                for f in data_dictionary.dataFields])

    def _nyoka_rule_set_model(self, rule_set_model: models.RuleSetModel) -> nyoka_pmml.RuleSetModel:
        return nyoka_pmml.RuleSetModel(
            functionName='classification',
            algorithmName='RuleSet',
            MiningSchema=None if rule_set_model.miningSchema is None else self._nyoka_mining_schema(
                rule_set_model.miningSchema),
            RuleSet=None if rule_set_model.ruleSet is None else self._nyoka_rule_set(rule_set_model.ruleSet))

    def _nyoka_mining_schema(self, mining_schema: models.MiningSchema) -> nyoka_pmml.MiningSchema:
        return nyoka_pmml.MiningSchema(
            MiningField=[
                nyoka_pmml.MiningField(name=f.name, usageType=f.usageType.name)
                for f in mining_schema.miningFields])

    def _nyoka_rule_set(self, rule_set: models.RuleSet) -> nyoka_pmml.RuleSet:
        return nyoka_pmml.RuleSet(
            RuleSelectionMethod=None if rule_set.ruleSelectionMethod is None else [
                nyoka_pmml.RuleSelectionMethod(criterion=m.name)
                for m in rule_set.ruleSelectionMethod],
            SimpleRule=None if rule_set.rules is None else [self._nyoka_rule(r) for r in rule_set.rules],
            recordCount=rule_set.recordCount,
            nbCorrect=rule_set.nbCorrect,
            defaultScore=rule_set.defaultScore,
            defaultConfidence=rule_set.defaultConfidence)

    def _nyoka_rule(self, simple_rule: models.SimpleRule) -> nyoka_pmml.SimpleRule:
        return nyoka_pmml.SimpleRule(
            SimplePredicate=None if (
                    simple_rule.predicate is None or not isinstance(simple_rule.predicate, models.SimplePredicate)) else
            nyoka_pmml.SimplePredicate(
                field=simple_rule.predicate.field,
                operator=simple_rule.predicate.operator.name,
                value=simple_rule.predicate.value),
            CompoundPredicate=None if (
                    simple_rule.predicate is None or not isinstance(simple_rule.predicate,
                                                                    models.CompoundPredicate)) else
            nyoka_pmml.CompoundPredicate(
                booleanOperator=simple_rule.predicate.booleanOperator.name,
                SimplePredicate=[
                    nyoka_pmml.SimplePredicate(field=sp.field, operator=sp.operator.name, value=sp.value)
                    for sp in simple_rule.predicate.simplePredicates]),
            score=simple_rule.score,
            id=simple_rule.id,
            recordCount=simple_rule.recordCount,
            nbCorrect=simple_rule.nbCorrect,
            confidence=simple_rule.confidence,
            weight=simple_rule.weight)

    def _nyoka_pmml_characteristics(self, characteristics: models.Characteristics) -> nyoka_pmml.Characteristics:
        return nyoka_pmml.Characteristics(
            Characteristic=[
                nyoka_pmml.Characteristic(
                    name=characteristic.name,
                    Attribute=[self._nyoka_pmml_attributes(attribute) for attribute in characteristic.attributes])
                for characteristic in characteristics.characteristics])

    def _nyoka_pmml_attributes(self, attribute: models.Attribute) -> nyoka_pmml.Attribute:
        return nyoka_pmml.Attribute(
            partialScore=attribute.score if not isinstance(attribute.score, models.ComplexPartialScore) else None,
            ComplexPartialScore=nyoka_pmml.ComplexPartialScore(
                Apply=nyoka_pmml.Apply(
                    function='+',
                    Apply_member=[nyoka_pmml.Apply(
                        function='*',
                        FieldRef=[nyoka_pmml.FieldRef(field=attribute.score.feature_name)],
                        Constant=[nyoka_pmml.Constant(valueOf_=attribute.score.multiplier)])],
                    Constant=[nyoka_pmml.Constant(valueOf_=attribute.score.constant)])) if isinstance(
                        attribute.score, models.ComplexPartialScore) else None,
            SimplePredicate=None if (attribute.predicate is None or not isinstance(
                attribute.predicate, models.SimplePredicate)) else nyoka_pmml.SimplePredicate(
                    field=attribute.predicate.field,
                    operator=attribute.predicate.operator.name,
                    value=attribute.predicate.value),
            CompoundPredicate=None if (attribute.predicate is None or not isinstance(
                attribute.predicate, models.CompoundPredicate)) else nyoka_pmml.CompoundPredicate(
                    booleanOperator=attribute.predicate.booleanOperator.name,
                    SimplePredicate=[
                        nyoka_pmml.SimplePredicate(field=sp.field, operator=sp.operator.name, value=sp.value)
                        for sp in attribute.predicate.simplePredicates]))
