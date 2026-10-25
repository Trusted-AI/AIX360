"""Full real TRXF Feature/Predicate modules; no algorithm or numeric mocks."""
from decimal import Decimal
from fractions import Fraction
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


def load_core():
    source_root = os.environ.get('AIX360_SOURCE_ROOT')
    if source_root is None:
        for parent in Path(__file__).resolve().parents:
            if (parent / 'aix360/algorithms/rule_induction/trxf/core/feature.py').is_file():
                source_root = str(parent)
                break
    if source_root is None:
        raise RuntimeError('Set AIX360_SOURCE_ROOT to the source snapshot')
    root = Path(source_root)
    packages = {}
    name = ''
    path = root
    for part in ('aix360', 'algorithms', 'rule_induction', 'trxf', 'core'):
        name = name + '.' + part if name else part
        path = path / part
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        packages[name] = package
    loaded = {}
    with patch.dict(sys.modules, packages):
        for leaf in ('boolean_evaluator', 'feature', 'predicate'):
            module_name = name + '.' + leaf
            spec = importlib.util.spec_from_file_location(module_name, path / (leaf + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded[leaf] = module
    return loaded['feature'].Feature, loaded['predicate'].Predicate, loaded['predicate'].Relation


Feature, Predicate, Relation = load_core()


class TestFeatureNumericOperands(unittest.TestCase):
    def test_fraction_divisor_is_one_operand(self):
        assignment = {'x': Fraction(1, 2), 'y': Fraction(1, 3)}
        self.assertEqual(Feature('x/y').evaluate(assignment), 1.5)

    def test_negative_fraction_divisor_is_one_operand(self):
        assignment = {'x': Fraction(1, 2), 'y': Fraction(-1, 3)}
        self.assertEqual(Feature('x/y').evaluate(assignment), -1.5)

    def test_fraction_division_in_compound_feature(self):
        assignment = {'x': Fraction(1, 2), 'y': Fraction(1, 3), 'z': 2}
        self.assertEqual(Feature('x/y + 2*z').evaluate(assignment), 5.5)

    def test_real_predicate_uses_correct_fraction_feature(self):
        assignment = {'x': Fraction(1, 2), 'y': Fraction(1, 3)}
        self.assertTrue(Predicate(Feature('x/y'), Relation.EQ, 1.5).evaluate(assignment))
        self.assertFalse(Predicate(Feature('x/y'), Relation.LT, 1).evaluate(assignment))

    def test_existing_numeric_precedence_and_operand_order(self):
        self.assertAlmostEqual(Feature('5+x*(4-y)').evaluate({'x': 6.3, 'y': 3.1}),
                               5 + 6.3 * (4 - 3.1))
        self.assertAlmostEqual(Feature('-x*(4-y)').evaluate({'x': 6.3, 'y': 3.1}),
                               -6.3 * (4 - 3.1))
        self.assertEqual(Feature('x-y').evaluate({'x': 6, 'y': 2}), 4)
        self.assertEqual(Feature('x/y').evaluate({'x': 6, 'y': 2}), 3)

    def test_existing_decimal_and_float_constant_combination(self):
        # Direct Decimal + float dispatch would break this previously accepted case.
        self.assertEqual(Feature('x+1').evaluate({'x': Decimal('1.5')}), 2.5)

    def test_categorical_string_and_standalone_boolean_remain_supported(self):
        self.assertEqual(Feature('category').evaluate({'category': 'foo'}), 'foo')
        self.assertEqual(Feature('status=A14').evaluate({'status=A14': True}), True)
        self.assertTrue(Predicate(Feature('category'), Relation.EQ, 'foo').evaluate({'category': 'foo'}))

    def test_arithmetic_boolean_and_string_remain_rejected(self):
        for value in (True, False, '2'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Feature('x+1').evaluate({'x': value})

    def test_zero_division_still_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Feature('x/y').evaluate({'x': 1, 'y': 0})


if __name__ == '__main__':
    unittest.main()
