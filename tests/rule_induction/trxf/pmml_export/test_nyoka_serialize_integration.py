import datetime
import io
from unittest import TestCase

try:
    import pypmml
    import pandas
except ImportError:
    pypmml = None
    pandas = None

import aix360.algorithms.rule_induction.trxf.pmml_export.models as models
import aix360.algorithms.rule_induction.trxf.pmml_export.serializer as serializer


class TestNyokaSerializer(TestCase):
    nyokaSerializer = serializer.NyokaSerializer(datetime.datetime.now())

    def setUp(self):
        if pypmml is None or pandas is None:
            self.skipTest('Install pypmml and pandas for integration tests')

    def test_serialize_then_predict(self):
        # arrange
        score_card = models.Scorecard(
            models.DataDictionary(
                [
                    models.DataField(
                        name='department', dataType=models.DataType.string, optype=models.OpType.categorical),
                    models.DataField(
                        name='age', dataType=models.DataType.integer, optype=models.OpType.continuous),
                    models.DataField(
                        name='income', dataType=models.DataType.double, optype=models.OpType.continuous),
                    models.DataField(
                        name='overallScore', dataType=models.DataType.double, optype=models.OpType.continuous)
                ]
            ),
            miningSchema=models.MiningSchema(
                [
                    models.MiningField(name='department', usageType=models.MiningFieldUsageType.active),
                    models.MiningField(name='age', usageType=models.MiningFieldUsageType.active),
                    models.MiningField(name='income', usageType=models.MiningFieldUsageType.active),
                    models.MiningField(name='overallScore', usageType=models.MiningFieldUsageType.target),
                ]
            ),
            output=models.Output([
                models.OutputField(
                    name='Final Score',
                    feature='predictedValue',
                    dataType=models.DataType.double,
                    optype=models.OpType.continuous)
            ]),
            characteristics=models.Characteristics(
                [
                    models.Characteristic(name='departmentScore', attributes=[
                        models.Attribute(
                            score='-9',
                            predicate=models.SimplePredicate(
                                field='department', operator=models.Operator.isMissing)),
                        models.Attribute(
                            score='19',
                            predicate=models.SimplePredicate(
                                field='department', operator=models.Operator.equal, value='marketing')),
                        models.Attribute(
                            score='3',
                            predicate=models.SimplePredicate(
                                field='department', operator=models.Operator.equal, value='engineering')),
                        models.Attribute(
                            score='6',
                            predicate=models.SimplePredicate(
                                field='department', operator=models.Operator.equal, value='business')),
                        models.Attribute(
                            score='0',
                            predicate=models.TruePredicate()),

                    ]),
                    models.Characteristic(
                        name='ageScore', attributes=[
                            models.Attribute(
                                score='-1',
                                predicate=models.SimplePredicate(
                                    field='age', operator=models.Operator.isMissing)),
                            models.Attribute(
                                score='-3',
                                predicate=models.SimplePredicate(
                                    field='age', operator=models.Operator.lessOrEqual, value='18')),
                            models.Attribute(
                                score='0',
                                predicate=models.CompoundPredicate(
                                    booleanOperator=models.BooleanOperator.and_,
                                    simplePredicates=[
                                        models.SimplePredicate(
                                            field='age', operator=models.Operator.greaterThan, value='18'),
                                        models.SimplePredicate(
                                            field='age', operator=models.Operator.lessOrEqual, value='29')])),
                            models.Attribute(
                                score='12',
                                predicate=models.CompoundPredicate(
                                    booleanOperator=models.BooleanOperator.and_,
                                    simplePredicates=[
                                        models.SimplePredicate(
                                            field='age', operator=models.Operator.greaterThan, value='29'),
                                        models.SimplePredicate(
                                            field='age', operator=models.Operator.lessOrEqual, value='39')])),
                            models.Attribute(
                                score='18',
                                predicate=models.SimplePredicate(
                                    field='age', operator=models.Operator.greaterThan, value='39'))]),
                    models.Characteristic(name='incomeScore', attributes=[
                        models.Attribute(
                            score='3',
                            predicate=models.SimplePredicate(
                                field='income', operator=models.Operator.isMissing)),
                        models.Attribute(
                            predicate=models.SimplePredicate(
                                field='income', operator=models.Operator.equal, value='1000'),
                            score=models.ComplexPartialScore(feature_name='income', multiplier='0.03', constant='11')),
                        models.Attribute(
                            score='5',
                            predicate=models.CompoundPredicate(
                                    booleanOperator=models.BooleanOperator.and_,
                                    simplePredicates=[
                                        models.SimplePredicate(
                                            field='income', operator=models.Operator.greaterThan, value='1000'),
                                        models.SimplePredicate(
                                            field='income', operator=models.Operator.lessOrEqual, value='2500')])),
                        models.Attribute(
                            predicate=models.SimplePredicate(
                                field='income', operator=models.Operator.greaterThan, value='1500'),
                            score=models.ComplexPartialScore(
                                feature_name='income', multiplier='0.01', constant='18'))])]))

        # when
        serialized = self.nyokaSerializer.serialize(score_card)
        pmml_model = pypmml.Model.load(io.StringIO(serialized))

        # assert
        self.assertIsNotNone(pmml_model)
        self.assertEqual(len(pmml_model.dataDictionary.fields), 4)
