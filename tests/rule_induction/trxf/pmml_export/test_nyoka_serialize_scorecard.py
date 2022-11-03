import datetime
import random
import unittest
import dataclasses

import xmltodict

import aix360.algorithms.rule_induction.trxf.pmml_export.models as models
import aix360.algorithms.rule_induction.trxf.pmml_export.serializer as serializer


class TestNyokaSerializer(unittest.TestCase):
    nyokaSerializer = serializer.NyokaSerializer(datetime.datetime.now())
    empty_scorecard = models.Scorecard(
        dataDictionary=models.DataDictionary(list()),
        miningSchema=models.MiningSchema(list()),
        output=models.Output(list()),
        initialScore="0",
        characteristics=models.Characteristics(list()))
    characteristics1 = models.Characteristics(
        characteristics=[models.Characteristic(
            name='age1',
            attributes=[models.Attribute(score='5', predicate=models.SimplePredicate(
                field='age1', operator=models.Operator.lessThan, value='18'))])])
    characteristics2 = models.Characteristics(
        characteristics=[models.Characteristic(
            name='age2',
            attributes=[models.Attribute(score='8', predicate=models.CompoundPredicate(
                [models.SimplePredicate(field='age2', operator=models.Operator.greaterThan, value='18'),
                 models.SimplePredicate(field='age2', operator=models.Operator.lessThan, value='25')],
                booleanOperator=models.BooleanOperator.and_))])])
    characteristics3 = models.Characteristics(
        characteristics=[models.Characteristic(
            name='type',
            attributes=[
                models.Attribute(score='4', predicate=models.SimplePredicate(
                    field='type', operator=models.Operator.equal, value='normal')),
                models.Attribute(score='6', predicate=models.SimplePredicate(
                    field='type', operator=models.Operator.equal, value='good')),
                models.Attribute(score='8', predicate=models.SimplePredicate(
                    field='type', operator=models.Operator.equal, value='very good'))])])
    characteristics4 = models.Characteristics(
        characteristics=[models.Characteristic(
            name='age4',
            attributes=[models.Attribute(
                predicate=models.SimplePredicate(field='age4', operator=models.Operator.lessThan, value='18'),
                score=models.ComplexPartialScore(feature_name='age4', multiplier='4', constant='3'))])])

    def test_serialize_scorecard_tag(self):
        # arrange
        initial_score = str(random.randint(25, 200))
        scorecard = dataclasses.replace(self.empty_scorecard, initialScore=initial_score)

        # when
        serialized = xmltodict.parse(self.nyokaSerializer.serialize(scorecard))
        scorecard_content = serialized['PMML']['Scorecard']

        # assert
        self.assertEqual(scorecard_content['@functionName'], 'regression')
        self.assertEqual(scorecard_content['@algorithmName'], 'ScoreCard')
        self.assertEqual(scorecard_content['@initialScore'], initial_score)

    def test_serialize_output_tag(self):
        # arrange
        scorecard = dataclasses.replace(self.empty_scorecard, output=models.Output(
            [
                models.OutputField(
                    name='Final Score',
                    feature='predictedValue',
                    dataType=models.DataType.double,
                    optype=models.OpType.continuous),
                models.OutputField(
                    name='Reason Code 1',
                    feature='reasonCode',
                    dataType=models.DataType.string,
                    optype=models.OpType.categorical)]))

        # when
        serialized = xmltodict.parse(TestNyokaSerializer.nyokaSerializer.serialize(scorecard))
        output_content = serialized['PMML']['Scorecard']['Output']['OutputField']

        # assert
        self.assertEqual(len(output_content), 2)
        self.assertEqual(output_content[0]['@name'], 'Final Score')
        self.assertEqual(output_content[0]['@feature'], 'predictedValue')
        self.assertEqual(output_content[0]['@dataType'], 'double')
        self.assertEqual(output_content[0]['@optype'], 'continuous')
        self.assertEqual(output_content[1]['@name'], 'Reason Code 1')
        self.assertEqual(output_content[1]['@feature'], 'reasonCode')
        self.assertEqual(output_content[1]['@dataType'], 'string')
        self.assertEqual(output_content[1]['@optype'], 'categorical')

    def test_serialize_characteristic(self):
        # arrange
        scorecard1 = dataclasses.replace(self.empty_scorecard, characteristics=self.characteristics1)
        scorecard2 = dataclasses.replace(self.empty_scorecard, characteristics=self.characteristics2)

        # when
        serialized1 = xmltodict.parse(TestNyokaSerializer.nyokaSerializer.serialize(scorecard1))
        serialized2 = xmltodict.parse(TestNyokaSerializer.nyokaSerializer.serialize(scorecard2))
        output_characteristic1 = serialized1['PMML']['Scorecard']['Characteristics']['Characteristic']
        output_characteristic2 = serialized2['PMML']['Scorecard']['Characteristics']['Characteristic']

        # assert
        self.assertEqual(output_characteristic1['@name'], 'age1')
        self.assertEqual(output_characteristic2['@name'], 'age2')

    def test_serialize_attribute_tag(self):
        # arrange
        scorecard1 = dataclasses.replace(self.empty_scorecard, characteristics=self.characteristics1)
        scorecard2 = dataclasses.replace(self.empty_scorecard, characteristics=self.characteristics2)

        # when
        serialized1 = xmltodict.parse(TestNyokaSerializer.nyokaSerializer.serialize(scorecard1))
        serialized2 = xmltodict.parse(TestNyokaSerializer.nyokaSerializer.serialize(scorecard2))
        attribute1_content = serialized1['PMML']['Scorecard']['Characteristics']['Characteristic']['Attribute']
        attribute2_content = serialized2['PMML']['Scorecard']['Characteristics']['Characteristic']['Attribute']

        # assert
        self.assertEqual(attribute1_content['@partialScore'], '5')
        self.assertEqual(attribute2_content['@partialScore'], '8')

    def test_serialize_closed_interval(self):
        # arrange
        # 18 < age < 25
        scorecard = dataclasses.replace(self.empty_scorecard, characteristics=self.characteristics2)

        # when
        serialized = xmltodict.parse(TestNyokaSerializer.nyokaSerializer.serialize(scorecard))
        interval_content = serialized[
            'PMML']['Scorecard']['Characteristics']['Characteristic']['Attribute']['CompoundPredicate']
        interval_lower = interval_content['SimplePredicate'][0]
        interval_upper = interval_content['SimplePredicate'][1]

        # assert
        self.assertEqual(interval_content['@booleanOperator'], 'and')
        self.assertEqual(interval_upper['@field'], 'age2')
        self.assertEqual(interval_upper['@operator'], 'lessThan')
        self.assertEqual(interval_upper['@value'], '25')
        self.assertEqual(interval_lower['@field'], 'age2')
        self.assertEqual(interval_lower['@operator'], 'greaterThan')
        self.assertEqual(interval_lower['@value'], '18')

    def test_serialize_half_opened_interval(self):
        # arrange
        # 18 < age
        scorecard = dataclasses.replace(self.empty_scorecard, characteristics=self.characteristics1)

        # when
        serialized = xmltodict.parse(TestNyokaSerializer.nyokaSerializer.serialize(scorecard))
        interval_content = serialized[
            'PMML']['Scorecard']['Characteristics']['Characteristic']['Attribute']
        interval_upper = interval_content['SimplePredicate']

        # assert
        self.assertEqual(interval_upper['@field'], 'age1')
        self.assertEqual(interval_upper['@operator'], 'lessThan')
        self.assertEqual(interval_upper['@value'], '18')

    def test_serialize_set(self):
        # arrange
        # types: very good, good, normal, bad
        scorecard = dataclasses.replace(self.empty_scorecard, characteristics=self.characteristics3)

        # when
        serialized = xmltodict.parse(TestNyokaSerializer.nyokaSerializer.serialize(scorecard))
        set_content = serialized['PMML']['Scorecard']['Characteristics']['Characteristic']['Attribute']

        # assert
        self.assertEqual(set_content[0]['SimplePredicate']['@field'], 'type')
        self.assertEqual(set_content[0]['SimplePredicate']['@operator'], 'equal')
        self.assertEqual(set_content[0]['SimplePredicate']['@value'], 'normal')
        self.assertEqual(set_content[1]['SimplePredicate']['@field'], 'type')
        self.assertEqual(set_content[1]['SimplePredicate']['@operator'], 'equal')
        self.assertEqual(set_content[1]['SimplePredicate']['@value'], 'good')
        self.assertEqual(set_content[2]['SimplePredicate']['@field'], 'type')
        self.assertEqual(set_content[2]['SimplePredicate']['@operator'], 'equal')
        self.assertEqual(set_content[2]['SimplePredicate']['@value'], 'very good')

    def test_serialize_complex_partial_score(self):
        # arrange
        # score = 4 * age4 + 3
        scorecard = dataclasses.replace(self.empty_scorecard, characteristics=self.characteristics4)

        # when
        serialized = xmltodict.parse(TestNyokaSerializer.nyokaSerializer.serialize(scorecard))
        score_content = serialized[
            'PMML']['Scorecard']['Characteristics']['Characteristic']['Attribute']['ComplexPartialScore']

        self.assertEqual(score_content['Apply']['@function'], '+')
        self.assertEqual(score_content['Apply']['Constant'], '3')
        self.assertEqual(score_content['Apply']['Apply']['@function'], '*')
        self.assertEqual(score_content['Apply']['Apply']['FieldRef']['@field'], 'age4')
        self.assertEqual(score_content['Apply']['Apply']['Constant'], '4')
