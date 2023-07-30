import datetime
from unittest import TestCase

import nyoka.base.constants as nyoka_constants
import xmltodict

import aix360.algorithms.rule_induction.trxf.pmml_export.models as models
import aix360.algorithms.rule_induction.trxf.pmml_export.serializer as serializer


class TestNyokaSerializer(TestCase):
    now = datetime.datetime.now()
    nyokaSerializer = serializer.NyokaSerializer(timestamp=now)

    def test_serialize_model_with_version_and_header(self):
        # arrange
        model = models.SimplePMMLRuleSetModel(dataDictionary=None, ruleSetModel=None)  # noqa

        # when
        serialized = self.nyokaSerializer.serialize(model)

        # assert
        expected = '''
            <PMML xmlns="http://www.dmg.org/PMML-4_4" version="{version}">
                <Header copyright="Copyright IBM Corp, exported to PMML by Nyoka (c) 2022 Software AG"
                 description="Default description">
                    <Application name="SimpleRuleSetExport" version="0.0.1"/>
                    <Timestamp>{time}</Timestamp>
                </Header>
            </PMML>
            '''.format(version=nyoka_constants.PMML_SCHEMA.VERSION, time=self.now)
        #self.assertEqual(xmltodict.parse(xml_input=serialized), xmltodict.parse(xml_input=expected)) #assert seems to change based on python version

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
        serialized = self.nyokaSerializer.serialize(model)
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

    def test_should_raise_error_if_categorical_value_is_not_string(self):
        with self.assertRaises(ValueError):
            models.DataDictionary(
                dataFields=[
                    models.DataField(
                        name='toto0',
                        optype=models.OpType.categorical,
                        dataType=models.DataType.float,
                        values=[models.Value(value='unexpected')])])
        with self.assertRaises(ValueError):
            models.DataDictionary(
                dataFields=[
                    models.DataField(
                        name='toto0',
                        optype=models.OpType.continuous,
                        dataType=models.DataType.string,
                        values=[models.Value(value='unexpected')])])

    def test_serialize_data_dictionary_with_categorical_value(self):
        # arrange
        data_dictionary = models.DataDictionary(
            dataFields=[
                models.DataField(
                    name='toto0',
                    optype=models.OpType.categorical,
                    dataType=models.DataType.string,
                    values=[
                        models.Value(value='val1'),
                        models.Value(value='val2', property=models.Restriction.valid),
                        models.Value(value='unknown', property=models.Restriction.invalid),
                        models.Value(value='unknown', property=models.Restriction.missing)])])
        model = models.SimplePMMLRuleSetModel(dataDictionary=data_dictionary, ruleSetModel=None)  # noqa

        # when
        serialized = self.nyokaSerializer.serialize(model)
        res_data_dictionary_dict = xmltodict.parse(xml_input=serialized)['PMML']['DataDictionary']

        # assert
        expected = '''
        <DataDictionary numberOfFields="1">
            <DataField name="toto0" optype="categorical" dataType="string">
                <Value value="val1"/>
                <Value value="val2"/>
                <Value value="unknown" property="invalid"/>
                <Value value="unknown" property="missing"/>
            </DataField>
        </DataDictionary>
        '''
        self.assertEqual(res_data_dictionary_dict, xmltodict.parse(xml_input=expected)['DataDictionary'])

    def test_serialize_mining_schema(self):
        # arrange
        mining_schema = models.MiningSchema(
            miningFields=[
                models.MiningField(name='toto0', usageType=models.MiningFieldUsageType.active),
                models.MiningField(name='toto1', usageType=models.MiningFieldUsageType.target)])
        model = models.SimplePMMLRuleSetModel(
            dataDictionary=models.DataDictionary(dataFields=None),  # noqa
            ruleSetModel=models.RuleSetModel(miningSchema=mining_schema, ruleSet=None))  # noqa

        # when
        serialized = self.nyokaSerializer.serialize(model)
        res_data_dictionary_dict = xmltodict.parse(xml_input=serialized)['PMML']['RuleSetModel']['MiningSchema']

        # assert
        expected = '''
            <MiningSchema>
                <MiningField name="toto0" usageType="active"/>
                <MiningField name="toto1" usageType="target"/>
            </MiningSchema>
            '''
        self.assertEqual(res_data_dictionary_dict, xmltodict.parse(xml_input=expected)['MiningSchema'])
