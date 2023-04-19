from aix360.algorithms.rule_induction.trxf.pmml_export import AbstractSerializer, AbstractReader


class PmmlExporter:
    def __init__(self, reader: AbstractReader, serializer: AbstractSerializer):
        self._serializer = serializer
        self._reader = reader

    def export(self, trxf_input):
        """
        Translate a given TRXF RuleSetClassifier or Scorecard to a PMML string
        @param trxf_input: A TRXF RuleSetClassifier or Scorecard object
        @return: The corresponding PMML string
        """
        if self._reader.data_dictionary is None:
            raise AssertionError("Missing data dictionary in reader object")
        return self._serializer.serialize(self._reader.read(trxf_input))
