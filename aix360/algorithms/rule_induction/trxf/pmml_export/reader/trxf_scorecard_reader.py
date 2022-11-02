import pandas as pd

from aix360.algorithms.rule_induction.trxf import scorecard
from aix360.algorithms.rule_induction.trxf.pmml_export import models
from aix360.algorithms.rule_induction.trxf.pmml_export.reader import AbstractReader
from aix360.algorithms.rule_induction.trxf.pmml_export.utilities import extract_data_dictionary, trxf_to_pmml_predicate


class TrxfScorecardReader(AbstractReader):
    def __init__(self, data_dictionary=None):
        self._data_dictionary = data_dictionary

    @property
    def data_dictionary(self):
        return self._data_dictionary

    def read(self, trxf_scorecard: scorecard.Scorecard) -> models.Scorecard:
        """
        Translate a TRXF Scorecard to an internal Scorecard
        """
        mining_schema = _extract_mining_schema(trxf_scorecard.features)
        output = models.Output([models.OutputField(name='RawResult',
                                                   feature='predictedValue',
                                                   dataType=models.DataType.double,
                                                   optype=models.OpType.continuous)])
        characteristics = _extract_characteristics(trxf_scorecard)

        assert self._data_dictionary is not None
        return models.Scorecard(dataDictionary=self._data_dictionary,
                                miningSchema=mining_schema,
                                output=output,
                                characteristics=characteristics,
                                initialScore=str(trxf_scorecard.bias))

    def load_data_dictionary(self, X: pd.DataFrame):
        """
        Extract the data dictionary from a feature dataframe, and store it
        """
        self._data_dictionary = extract_data_dictionary(X)


def _extract_mining_schema(scorecard_features):
    mining_fields = {}
    for feature in scorecard_features:
        name = feature.variable_names[0]
        if name not in mining_fields:
            mining_field = models.MiningField(name=name)
            mining_fields[name] = mining_field
    return models.MiningSchema(miningFields=list(mining_fields.values()))


def _extract_characteristics(trxf_scorecard):
    characteristics = []
    for partition in trxf_scorecard.partitions:
        feature_name = partition.feature.variable_names[0]
        attributes = []
        for bin in partition.bins:
            assert isinstance(bin, scorecard.IntervalBin), "Scorecard is only supported for continuous bins"
            conjunction = bin.to_conjunction()
            predicate = trxf_to_pmml_predicate(conjunction)
            score = str(bin.sub_score)
            attribute = models.Attribute(score=score, predicate=predicate)
            attributes.append(attribute)
        characteristic = models.Characteristic(name=feature_name, attributes=attributes)
        characteristics.append(characteristic)
    return models.Characteristics(characteristics)


