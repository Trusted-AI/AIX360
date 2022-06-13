from unittest import TestCase
from unittest.mock import MagicMock
from aix360.algorithms.rule_induction.trxf.core import Feature
import aix360.algorithms.rule_induction.trxf.scorecard as sc


class TestLinearIntervalBin(TestCase):
    def test_construct_with_missing_endpoints(self):
        mock_feature = Feature('feature')

        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0)
        self.assertEqual(int_bin.left_end, float('-inf'))
        self.assertEqual(int_bin.right_end, float('inf'))

        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, left_end=1.0)
        self.assertEqual(int_bin.left_end, 1.0)
        self.assertEqual(int_bin.right_end, float('inf'))

        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, right_end=0.0)
        self.assertEqual(int_bin.left_end, float('-inf'))
        self.assertEqual(int_bin.right_end, 0.0)

        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, left_end=1.0, right_end=2.0)
        self.assertEqual(int_bin.left_end, 1.0)
        self.assertEqual(int_bin.right_end, 2.0)

    def test_construct_with_invalid_endpoints(self):
        mock_feature = Feature('feature')
        self.assertRaises(ValueError, sc.LinearIntervalBin, mock_feature, 0.0, 0.0, left_end=float('inf'))
        self.assertRaises(ValueError, sc.LinearIntervalBin, mock_feature, 0.0, 0.0, right_end=float('-inf'))
        self.assertRaises(ValueError, sc.LinearIntervalBin, mock_feature, 0.0, 0.0, 1.0, -1.0)
        self.assertRaises(ValueError, sc.LinearIntervalBin, mock_feature, 0.0, 0.0, 0.0, 0.0)

    def test_contains(self):
        assignment = {'feature': 1.0}

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=1.0)
        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        self.assertFalse(int_bin.contains(assignment))

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.0)
        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        self.assertTrue(int_bin.contains(assignment))

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=-1.0)
        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        self.assertFalse(int_bin.contains(assignment))

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=2.0)
        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        self.assertFalse(int_bin.contains(assignment))

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.7)
        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        self.assertTrue(int_bin.contains(assignment))

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=float('inf'))
        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0)
        self.assertFalse(int_bin.contains(assignment))

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=float('-inf'))
        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0)
        self.assertTrue(int_bin.contains(assignment))

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=10.0)
        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0)
        self.assertTrue(int_bin.contains(assignment))

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value='non_numerical_value')
        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0)
        self.assertRaises(ValueError, int_bin.contains, assignment)

    def test_overlaps(self):
        mock_feature = Feature('feature')

        bin1 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        bin2 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        self.assertTrue(bin1.overlaps(bin2))

        bin1 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        bin2 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -1.0, 0.0)
        self.assertFalse(bin1.overlaps(bin2))

        bin1 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        bin2 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 1.0, 2.0)
        self.assertFalse(bin1.overlaps(bin2))

        bin1 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        bin2 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.5, 2.0)
        self.assertTrue(bin1.overlaps(bin2))

        bin1 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        bin2 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -0.5, 0.5)
        self.assertTrue(bin1.overlaps(bin2))

        bin1 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        bin2 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -0.5, 2.0)
        self.assertTrue(bin1.overlaps(bin2))

        bin1 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        bin2 = sc.IntervalBin(mock_feature, 0.0, -0.5, 2.0)
        self.assertTrue(bin1.overlaps(bin2))

        bin1 = sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
        bin2 = sc.SetBin(mock_feature, 0.0, {0.0, 1.0})
        self.assertRaises(ValueError, bin1.overlaps, bin2)
    
    def test_evaluate(self):
        assignment = {'feature': 1.0}

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=1.0)
        int_bin = sc.LinearIntervalBin(mock_feature, -1.0, 0.5, 0.0, 1.0)
        self.assertEqual(int_bin.evaluate(assignment), 0.0)

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.5)
        int_bin = sc.LinearIntervalBin(mock_feature, -1.0, 0.5, 0.0, 1.0)
        self.assertEqual(int_bin.evaluate(assignment), -0.75)

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.0)
        int_bin = sc.LinearIntervalBin(mock_feature, -1.0, 0.5, 0.0, 1.0)
        self.assertEqual(int_bin.evaluate(assignment), -1.0)

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value='non_numerical_value')
        int_bin = sc.LinearIntervalBin(mock_feature, 0.0, 0.0)
        self.assertRaises(ValueError, int_bin.evaluate, assignment)


class TestSetBin(TestCase):
    def test_construct_with_empty_values(self):
        mock_feature = Feature('feature')
        self.assertRaises(ValueError, sc.SetBin, mock_feature, 0.0, {})

        mock_feature = Feature('feature')
        set_bin = sc.SetBin(mock_feature, 0.0, {0, 1})
        self.assertSetEqual(set_bin.values, {0, 1})

    def test_contains(self):
        assignment = {'feature': 1.0}

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=1.0)
        set_bin = sc.SetBin(mock_feature, 0.0, {0.0})
        self.assertFalse(set_bin.contains(assignment))

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.0)
        set_bin = sc.SetBin(mock_feature, 0.0, {0.0})
        self.assertTrue(set_bin.contains(assignment))

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.5)
        set_bin = sc.SetBin(mock_feature, 0.0, {0.0, 1.0})
        self.assertFalse(set_bin.contains(assignment))

    def test_overlaps(self):
        mock_feature = Feature('feature')

        bin1 = sc.SetBin(mock_feature, 0.0, {0.0})
        bin2 = sc.SetBin(mock_feature, 0.0, {1.0})
        self.assertFalse(bin1.overlaps(bin2))

        bin1 = sc.SetBin(mock_feature, 0.0, {0.0, 1.0})
        bin2 = sc.SetBin(mock_feature, 0.0, {1.0})
        self.assertTrue(bin1.overlaps(bin2))

        bin1 = sc.SetBin(mock_feature, 0.0, {1.0})
        bin2 = sc.SetBin(mock_feature, 0.0, {-1.0, 1.0})
        self.assertTrue(bin1.overlaps(bin2))

        bin1 = sc.SetBin(mock_feature, 0.0, {0.0})
        bin2 = sc.SetBin(mock_feature, 0.0, {-1.0, 1.0})
        self.assertFalse(bin1.overlaps(bin2))

        bin1 = sc.SetBin(mock_feature, 0.0, {0.0, 1.0})
        bin2 = sc.SetBin(mock_feature, 0.0, {0.0, 1.0})
        self.assertTrue(bin1.overlaps(bin2))

    def test_evaluate(self):
        assignment = {'feature': 0.0}

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=2.0)
        int_bin = sc.SetBin(mock_feature, -1.0, {0.0, 1.0})
        self.assertEqual(int_bin.evaluate(assignment), 0.0)

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=1.0)
        int_bin = sc.SetBin(mock_feature, -1.0, {0.0, 1.0})
        self.assertEqual(int_bin.evaluate(assignment), -1.0)


class TestIntervalBin(TestCase):
    def test_zero_linear_multiplier(self):
        mock_feature = Feature('feature')
        int_bin = sc.IntervalBin(mock_feature, 1.0, 2.0, 3.0)
        self.assertEqual(int_bin.linear_multiplier, 0.0)

    def test_evaluate(self):
        assignment = {'feature': 1.0}

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=1.0)
        int_bin = sc.IntervalBin(mock_feature, -1.0, 0.0, 1.0)
        self.assertEqual(int_bin.evaluate(assignment), 0.0)

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.0)
        int_bin = sc.IntervalBin(mock_feature, -1.0, 0.0, 1.0)
        self.assertEqual(int_bin.evaluate(assignment), -1.0)

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.5)
        int_bin = sc.IntervalBin(mock_feature, -1.0, 0.0, 1.0)
        self.assertEqual(int_bin.evaluate(assignment), -1.0)
