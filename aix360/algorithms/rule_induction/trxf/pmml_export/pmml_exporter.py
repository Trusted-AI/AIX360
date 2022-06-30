from aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier import RuleSetClassifier
from aix360.algorithms.rule_induction.trxf.pmml_export import AbstractSerializer, AbstractReader


class PmmlExporter:
    def __init__(self, reader: AbstractReader, serializer: AbstractSerializer):
        self._serializer = serializer
        self._reader = reader

    def export(self, trxf_classifier: RuleSetClassifier):
        """
        Translate a given TRXF RuleSetClassifier to a PMML string
        @param trxf_classifier: A TRXF RuleSetClassifier
        @return: The corresponding PMML string
        """
        if self._reader.data_dictionary is None:
            raise AssertionError("Missing data dictionary in reader object")
        return self._serializer.serialize(self._reader.read(trxf_classifier))
