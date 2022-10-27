from unittest import TestCase

import xmltodict

import aix360.algorithms.rule_induction.trxf.pmml_export.models as models
import aix360.algorithms.rule_induction.trxf.pmml_export.serializer as serializer


class TestNyokaSerializer(TestCase):
    def test_serialize_rule_set(self):
        # arrange
        srz = serializer.NyokaSerializer()
        model = models.SimplePMMLRuleSetModel(
            dataDictionary=models.DataDictionary(dataFields=None),  # noqa
            ruleSetModel=models.RuleSetModel(
                miningSchema=None,  # noqa
                ruleSet=models.RuleSet(
                    ruleSelectionMethod=[models.RuleSelectionMethod.firstHit],
                    rules=None,  # noqa
                    recordCount=5,
                    nbCorrect=3,
                    defaultScore="toto",
                    defaultConfidence=0.5)))
        model_default = models.SimplePMMLRuleSetModel(
            dataDictionary=models.DataDictionary(dataFields=None),  # noqa
            ruleSetModel=models.RuleSetModel(
                miningSchema=None,  # noqa
                ruleSet=models.RuleSet(
                    ruleSelectionMethod=[models.RuleSelectionMethod.firstHit],
                    rules=None)))  # noqa

        # when
        serialized = srz.serialize(model)
        serialized_default = srz.serialize(model_default)
        res_data_dictionary_dict = xmltodict.parse(xml_input=serialized)['PMML']['RuleSetModel']['RuleSet']
        res_data_dictionary_dict_default = xmltodict.parse(xml_input=serialized_default)['PMML']['RuleSetModel'][
            'RuleSet']

        # assert
        expected = '''
            <RuleSet recordCount="5" nbCorrect="3" defaultScore="toto" defaultConfidence="0.5">
                <RuleSelectionMethod criterion="firstHit"/>
            </RuleSet>
            '''
        expected_default = '''
                <RuleSet>
                    <RuleSelectionMethod criterion="firstHit"/>
                </RuleSet>
                '''
        self.assertEqual(res_data_dictionary_dict, xmltodict.parse(xml_input=expected)['RuleSet'])
        self.assertEqual(res_data_dictionary_dict_default, xmltodict.parse(xml_input=expected_default)['RuleSet'])

    def test_serialize_predicate(self):
        # arrange
        srz = serializer.NyokaSerializer()
        model = models.SimplePMMLRuleSetModel(
            dataDictionary=models.DataDictionary(dataFields=None),  # noqa
            ruleSetModel=models.RuleSetModel(
                miningSchema=None,  # noqa
                ruleSet=models.RuleSet(
                    ruleSelectionMethod=None,  # noqa
                    rules=[models.SimpleRule(
                        score='test1',
                        predicate=models.SimplePredicate(
                            field='toto1', operator=models.Operator.greaterOrEqual, value='128')),
                        models.SimpleRule(
                            score='test2',
                            predicate=models.CompoundPredicate(
                                booleanOperator=models.BooleanOperator.and_,
                                simplePredicates=[
                                    models.SimplePredicate(field='toto2', operator=models.Operator.lessThan,
                                                           value='20.5'),
                                    models.SimplePredicate(field='toto2', operator=models.Operator.equal,
                                                           value='good')]),
                            id='test-id',
                            recordCount=5,
                            nbCorrect=3,
                            confidence=0.86,
                            weight=0.6)
                    ])))

        # when
        serialized = srz.serialize(model)
        res_data_dictionary_dict = xmltodict.parse(xml_input=serialized)['PMML']['RuleSetModel']['RuleSet']

        # assert
        expected = '''
            <RuleSet>
                <SimpleRule score="test1">
                    <SimplePredicate field="toto1" operator="greaterOrEqual" value="128"/>
                </SimpleRule>
                <SimpleRule id="test-id" score="test2" recordCount="5" nbCorrect="3" confidence="0.86" weight="0.6">
                    <CompoundPredicate booleanOperator="and">
                        <SimplePredicate field="toto2" operator="lessThan" value="20.5"/>
                        <SimplePredicate field="toto2" operator="equal" value="good"/>
                    </CompoundPredicate>
                </SimpleRule>
            </RuleSet>
            '''
        self.assertEqual(res_data_dictionary_dict, xmltodict.parse(xml_input=expected)['RuleSet'])
