import tokenize
from unittest import TestCase

from aix360.algorithms.rule_induction.trxf.core.feature import Feature, _shunting_yard


class TestFeature(TestCase):
    def test_variable_names(self):
        feature = Feature('x1.val * 3.2 + 1e2* x2')
        actual = feature.variable_names
        expected = ['x1.val', 'x2']
        self.assertListEqual(actual, expected)

    def test_evaluate_numerical(self):
        feature = Feature('   5 +x1 *    (4-x2)')
        actual = feature.evaluate({'x1': 6.3, 'x2': 3.1})
        expected = 5 + 6.3 * (4 - 3.1)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_numerical_starts_with_unary_minus(self):
        feature = Feature('   -x1 *    (4-x2)')
        actual = feature.evaluate({'x1': 6.3, 'x2': 3.1})
        expected = -6.3 * (4 - 3.1)
        self.assertAlmostEqual(actual, expected)

    def test_evaluate_illegal_assignment_should_raise(self):
        feature = Feature('   5 +x1 *    (4-x2)')
        self.assertRaises(ValueError, feature.evaluate, {'x1': 6.3, 'x2': True})
        self.assertRaises(ValueError, feature.evaluate, {'x1': 6.3, 'x2': 'True'})
        self.assertRaises(ValueError, feature.evaluate, {'x1': 6.3})

    def test_evaluate_categorical(self):
        feature = Feature('x1')
        actual = feature.evaluate({'x1': 'foo'})
        expected = 'foo'
        self.assertEqual(actual, expected)

        actual = feature.evaluate({'x1': True})
        expected = True
        self.assertEqual(actual, expected)

    def test_malformed_expression_should_raise(self):
        missing_op = '   5 +x1     (4-x2)'
        self.assertRaises(ValueError, Feature, missing_op)

        empty_expression = ''
        self.assertRaises(ValueError, Feature, empty_expression)

        unmatched_parenthesis = '   5 +x1  *  (4-x2))'
        self.assertRaises(tokenize.TokenError, Feature, unmatched_parenthesis)

    def test_special_character_in_feature_name(self):
        feature = Feature('status=A14')
        actual = feature.evaluate({'status=A14': True})
        self.assertTrue(actual)
        feature = Feature('status==A14')
        actual = feature.evaluate({'status==A14': True})
        self.assertTrue(actual)
        feature = Feature('status.A14')
        actual = feature.evaluate({'status.A14': True})
        self.assertTrue(actual)
        feature = Feature('status%A14')
        actual = feature.evaluate({'status%A14': True})
        self.assertTrue(actual)
        feature = Feature('status^A14')
        actual = feature.evaluate({'status^A14': True})
        self.assertTrue(actual)
        feature = Feature('.status^A14')
        actual = feature.evaluate({'.status^A14': True})
        self.assertTrue(actual)

    def test_shunting_yard(self):
        actual = _shunting_yard('    x1 +4.2 * ( x2 - 1) ')
        expected = ['x1', '4.2', 'x2', '1', '-', '*', '+']
        self.assertListEqual(actual, expected)


