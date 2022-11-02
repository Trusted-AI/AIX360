from unittest import TestCase

import aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier as classifier
import aix360.algorithms.rule_induction.trxf.scorecard as sc
from aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier import RuleSetClassifier
from aix360.algorithms.rule_induction.trxf.pmml_export.models import RuleSetModel, SimplePMMLRuleSetModel
from aix360.algorithms.rule_induction.trxf.pmml_export.reader.trxf_ruleset_reader import TrxfRuleSetReader
from aix360.algorithms.rule_induction.trxf.pmml_export.reader.trxf_scorecard_reader import TrxfScorecardReader
from tests.rule_induction.trxf.utilities import create_test_ruleset, DATA_DICTIONARY, TEST_MINING_SCHEMA, \
    TEST_PMML_RULESET, DATA_FRAME, create_test_scorecard, PARTITIONS, DATA_DICTIONARY_SC


class TestTrxfReader(TestCase):
    def test_read_ruleset(self):
        reader = TrxfRuleSetReader(DATA_DICTIONARY)
        test_ruleset = create_test_ruleset()
        test_classifier = RuleSetClassifier([test_ruleset], classifier.RuleSelectionMethod.FIRST_HIT, default_label=0)
        ruleset_model = RuleSetModel(miningSchema=TEST_MINING_SCHEMA, ruleSet=TEST_PMML_RULESET)
        expected = SimplePMMLRuleSetModel(dataDictionary=DATA_DICTIONARY, ruleSetModel=ruleset_model)
        self.assertEqual(reader.read(test_classifier), expected)

    def test_load_data_dictionary(self):
        reader = TrxfRuleSetReader()
        reader.load_data_dictionary(DATA_FRAME)
        self.assertEqual(reader.data_dictionary, DATA_DICTIONARY)

    def test_read_scorecard(self):
        expected = create_test_scorecard()
        scorecard = sc.Scorecard(PARTITIONS, bias=100)
        reader = TrxfScorecardReader(DATA_DICTIONARY_SC)
        actual = reader.read(scorecard)
        self.assertEqual(expected, actual)
