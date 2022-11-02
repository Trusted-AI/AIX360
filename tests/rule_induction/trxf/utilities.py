import numpy as np
import pandas as pd

import aix360.algorithms.rule_induction.trxf.scorecard as sc
from aix360.algorithms.rule_induction.trxf.core import Predicate, Feature, Relation, Conjunction, DnfRuleSet
from aix360.algorithms.rule_induction.trxf.pmml_export import models
from aix360.algorithms.rule_induction.trxf.pmml_export.models import DataDictionary, DataField, OpType, DataType, \
    MiningSchema, MiningField, MiningFieldUsageType, RuleSet, SimpleRule, CompoundPredicate, SimplePredicate, Operator, \
    BooleanOperator, Output, OutputField, Attribute, Characteristic, Characteristics, Scorecard

DATA_DICTIONARY = DataDictionary(
    dataFields=[DataField(name='toto0', optype=OpType.continuous, dataType=DataType.double),
                DataField(name='toto1', optype=OpType.categorical, dataType=DataType.string, values=[Value('foo')]),
                DataField(name='toto2', optype=OpType.categorical, dataType=DataType.boolean),
                DataField(name='toto3', optype=OpType.ordinal, dataType=DataType.integer)]
)

DATA_FRAME = pd.DataFrame({'toto0': [np.double(1.2)],
                           'toto1': ['foo'],
                           'toto2': [False],
                           'toto3': [2]})

TEST_MINING_SCHEMA = MiningSchema(
    [MiningField(name='toto0', usageType=MiningFieldUsageType.active),
     MiningField(name='toto2', usageType=MiningFieldUsageType.active),
     MiningField(name='toto1', usageType=MiningFieldUsageType.active),
     MiningField(name='toto3', usageType=MiningFieldUsageType.active)]
)

TEST_PMML_RULESET = RuleSet(ruleSelectionMethod=[models.RuleSelectionMethod.firstHit],
                            rules=[
                                SimpleRule(
                                    predicate=CompoundPredicate(
                                        [SimplePredicate(operator=Operator.lessThan, value='0.1', field='toto0'),
                                         SimplePredicate(operator=Operator.equal, value='False', field='toto2')],
                                        booleanOperator=BooleanOperator.and_),
                                    score='1',
                                    id='[toto0 < 0.1] ^ [toto2 == False]'),
                                SimpleRule(
                                    predicate=CompoundPredicate(
                                        [SimplePredicate(operator=Operator.equal, value='foo', field='toto1'),
                                         SimplePredicate(operator=Operator.greaterOrEqual, value='1',
                                                         field='toto3')],
                                        booleanOperator=BooleanOperator.and_),
                                    score='1',
                                    id='[toto1 == foo] ^ [toto3 >= 1]'),
                                SimpleRule(
                                    predicate=CompoundPredicate(
                                        [SimplePredicate(operator=Operator.equal, value='foo', field='toto1'),
                                         SimplePredicate(operator=Operator.equal, value='False', field='toto2')],
                                        booleanOperator=BooleanOperator.and_),
                                    score='1',
                                    id='[toto1 == foo] ^ [toto2 == False]'),
                                SimpleRule(
                                    predicate=CompoundPredicate(
                                        [SimplePredicate(operator=Operator.lessThan, value='0.1', field='toto0'),
                                         SimplePredicate(operator=Operator.equal, value='foo', field='toto1'),
                                         SimplePredicate(operator=Operator.equal, value='False', field='toto2'),
                                         SimplePredicate(operator=Operator.greaterOrEqual, value='1',
                                                         field='toto3')],
                                        booleanOperator=BooleanOperator.and_),
                                    score='1',
                                    id='[toto0 < 0.1] ^ [toto1 == foo] ^ [toto2 == False] ^ [toto3 >= 1]')
                            ],
                            defaultScore='0')


def create_numerical_test_data(n_rows):
    np.random.seed(1)
    X = pd.DataFrame(np.random.randn(n_rows, 4), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.binomial(1, 0.4, n_rows))
    return X, y


def create_numerical_test_ruleset_pos():
    p1 = Predicate(Feature('x1'), Relation.LT, 0)
    p2 = Predicate(Feature('x2'), Relation.LT, 0)
    p3 = Predicate(Feature('x3'), Relation.LT, 0)
    p4 = Predicate(Feature('x4'), Relation.LT, 0)
    c1 = Conjunction([p1, p3])
    c2 = Conjunction([p2, p4])
    c3 = Conjunction([p2, p3])
    c4 = Conjunction([p1, p2, p3, p4])
    ruleset = DnfRuleSet([c1, c2, c3, c4], then_part=1)
    return ruleset


def create_numerical_test_ruleset_neg():
    p1 = Predicate(Feature('x1'), Relation.LT, 0)
    p2 = Predicate(Feature('x2'), Relation.LT, 0)
    p3 = Predicate(Feature('x3'), Relation.LT, 0)
    p4 = Predicate(Feature('x4'), Relation.LT, 0)
    c1 = Conjunction([-p1, -p2])
    c2 = Conjunction([-p2, -p3])
    c3 = Conjunction([-p1, -p4])
    c4 = Conjunction([-p4, -p3])
    ruleset = DnfRuleSet([c1, c2, c3, c4], then_part=0)
    return ruleset


def create_test_ruleset():
    p1 = Predicate(Feature('toto0'), Relation.LT, 0.1)
    p2 = Predicate(Feature('toto1'), Relation.EQ, 'foo')
    p3 = Predicate(Feature('toto2'), Relation.EQ, False)
    p4 = Predicate(Feature('toto3'), Relation.GE, 1)
    c1 = Conjunction([p1, p3])
    c2 = Conjunction([p2, p4])
    c3 = Conjunction([p2, p3])
    c4 = Conjunction([p1, p2, p3, p4])
    ruleset = DnfRuleSet([c1, c2, c3, c4], then_part=1)
    return ruleset


PARTITIONS = [
    sc.Partition([
        sc.IntervalBin(Feature('feature'), 5, None, 0.0),
        sc.IntervalBin(Feature('feature'), -10, 0.0, 1.0),
        sc.IntervalBin(Feature('feature'), 10, 1.0, None)
    ])
]
DATA_DICTIONARY_SC = DataDictionary(
    dataFields=[DataField(name='feature', optype=OpType.continuous, dataType=DataType.double)]
)

MINING_SCHEMA_SC = MiningSchema(
    [MiningField(name='feature', usageType=MiningFieldUsageType.active)]
)
OUTPUT = Output([OutputField('RawResult', 'predictedValue', DataType.double, OpType.continuous)])


def create_test_scorecard():
    predicate1 = CompoundPredicate([SimplePredicate(operator=Operator.lessThan, value='0.0', field='feature')],
                                   booleanOperator=BooleanOperator.and_)
    attribute1 = Attribute('5', predicate1)
    predicate2 = CompoundPredicate(
        [SimplePredicate(operator=Operator.greaterOrEqual, value='0.0', field='feature'),
         SimplePredicate(operator=Operator.lessThan, value='1.0', field='feature')],
        booleanOperator=BooleanOperator.and_)
    attribute2 = Attribute('-10', predicate2)
    predicate3 = CompoundPredicate(
        [SimplePredicate(operator=Operator.greaterOrEqual, value='1.0', field='feature')],
        booleanOperator=BooleanOperator.and_)
    attribute3 = Attribute('10', predicate3)
    characteristic = Characteristic('feature', [attribute1, attribute2, attribute3])
    characteristics = Characteristics([characteristic])
    return Scorecard(DATA_DICTIONARY_SC, MINING_SCHEMA_SC, OUTPUT, characteristics, '100')
