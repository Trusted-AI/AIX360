from unittest import TestCase

import nyoka.base.constants as nyoka_constants
import xmltodict

import aix360.algorithms.rule_induction.trxf.pmml_export.models as models
import aix360.algorithms.rule_induction.trxf.pmml_export.serializer as serializer


class TestNyokaSerializer(TestCase):
    def test_serialize_model_with_version_and_header(self):
        # arrange
        srz = serializer.NyokaSerializer()
        model = models.SimplePMMLRuleSetModel(dataDictionary=None, ruleSetModel=None)  # noqa

        # when
        serialized = srz.serialize(model)
        res_dict_no_timestamp = xmltodict.parse(xml_input=serialized)
        res_dict_no_timestamp['PMML']['Header'] = {
            k: res_dict_no_timestamp['PMML']['Header'][k]
            for k in res_dict_no_timestamp['PMML']['Header'] if k != 'Timestamp'}

        # assert
        expected = '''
            <PMML xmlns="http://www.dmg.org/PMML-4_4" version="{version}">
                <Header copyright="Copyright IBM Corp, exported to PMML by Nyoka (c) 2022 Software AG"
                 description="Default description">
                    <Application name="SimpleRuleSetExport" version="0.0.1"/>
                </Header>
            </PMML>
            '''.format(version=nyoka_constants.PMML_SCHEMA.VERSION)
        self.assertEqual(res_dict_no_timestamp, xmltodict.parse(xml_input=expected))

    def test_serialize_data_dictionary(self):
        # arrange
        srz = serializer.NyokaSerializer()
        data_dictionary = models.DataDictionary(
            dataFields=[
                models.DataField(name='toto0', optype=models.OpType.continuous, dataType=models.DataType.float),
                models.DataField(name='toto1', optype=models.OpType.ordinal, dataType=models.DataType.string),
                models.DataField(name='toto2', optype=models.OpType.categorical, dataType=models.DataType.boolean),
                models.DataField(name='toto3', optype=models.OpType.categorical, dataType=models.DataType.integer)])
        model = models.SimplePMMLRuleSetModel(dataDictionary=data_dictionary, ruleSetModel=None)  # noqa

        # when
        serialized = srz.serialize(model)
        res_data_dictionary_dict = xmltodict.parse(xml_input=serialized)['PMML']['DataDictionary']

        # assert
        expected = '''
        <DataDictionary numberOfFields="4">
            <DataField name="toto0" optype="continuous" dataType="float"/>
            <DataField name="toto1" optype="ordinal" dataType="string"/>
            <DataField name="toto2" optype="categorical" dataType="boolean"/>
            <DataField name="toto3" optype="categorical" dataType="integer"/>
        </DataDictionary>
        '''
        self.assertEqual(res_data_dictionary_dict, xmltodict.parse(xml_input=expected)['DataDictionary'])

    def test_serialize_mining_schema(self):
        # arrange
        srz = serializer.NyokaSerializer()
        mining_schema = models.MiningSchema(
            miningFields=[
                models.MiningField(name='toto0', usageType=models.MiningFieldUsageType.active),
                models.MiningField(name='toto1', usageType=models.MiningFieldUsageType.target)])
        model = models.SimplePMMLRuleSetModel(
            dataDictionary=models.DataDictionary(dataFields=None),  # noqa
            ruleSetModel=models.RuleSetModel(miningSchema=mining_schema, ruleSet=None))  # noqa

        # when
        serialized = srz.serialize(model)
        res_data_dictionary_dict = xmltodict.parse(xml_input=serialized)['PMML']['RuleSetModel']['MiningSchema']

        # assert
        expected = '''
            <MiningSchema>
                <MiningField name="toto0" usageType="active"/>
                <MiningField name="toto1" usageType="target"/>
            </MiningSchema>
            '''
        self.assertEqual(res_data_dictionary_dict, xmltodict.parse(xml_input=expected)['MiningSchema'])

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
