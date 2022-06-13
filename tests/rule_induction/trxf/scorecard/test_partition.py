from unittest import TestCase
from unittest.mock import MagicMock
import aix360.algorithms.rule_induction.trxf.scorecard as sc
from aix360.algorithms.rule_induction.trxf.core import Feature


class TestPartition(TestCase):

    def test_empty_bins(self):
        self.assertRaises(ValueError, sc.Partition, [])

    def test_bin_type_variation(self):
        mock_feature = Feature('feature')
        mock_bins = [
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -1.0, 0.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 1.0, 2.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 2.0, 3.0)
        ]
        partition = sc.Partition(mock_bins)
        self.assertEqual(partition.num_bins, 4)

        mock_feature = Feature('feature')
        mock_bins = [
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -1.0, 0.0),
            sc.SetBin(mock_feature, 0.0, {0.0, 0.5}),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 1.0, 2.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 2.0, 3.0)
        ]
        self.assertRaises(ValueError, sc.Partition, mock_bins)

        mock_feature = Feature('feature')
        mock_bins = [
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -1.0, 0.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 1.0, 2.0),
            sc.IntervalBin(mock_feature, 0.0, 0.0, 1.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 2.0, 3.0)
        ]
        self.assertRaises(ValueError, sc.Partition, mock_bins)

    def test_feature_mismatch_violation(self):
        mock_feature = Feature('feature')
        mock_bins = [
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -1.0, 0.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 1.0, 2.0)
        ]
        partition = sc.Partition(mock_bins)
        self.assertEqual(partition.feature, mock_feature)

        mock_feature1 = Feature('feature1')
        mock_feature2 = Feature('feature2')
        mock_bins = [
            sc.LinearIntervalBin(mock_feature1, 0.0, 0.0, -1.0, 0.0),
            sc.LinearIntervalBin(mock_feature2, 0.0, 0.0, 0.0, 1.0),
            sc.LinearIntervalBin(mock_feature1, 0.0, 0.0, 1.0, 2.0)
        ]
        self.assertRaises(ValueError, sc.Partition, mock_bins)

    def test_bin_overlap_violation(self):
        mock_feature = Feature('feature')
        mock_bins = [
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -1.0, 0.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -0.1, 1.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 1.0, 2.0)
        ]
        self.assertRaises(ValueError, sc.Partition, mock_bins)

        mock_feature = Feature('feature')
        mock_bins = [
            sc.SetBin(mock_feature, 0.0, {-1.0, 0.0}),
            sc.SetBin(mock_feature, 0.0, {-0.1, 1.1}),
            sc.SetBin(mock_feature, 0.0, {1.0, 2.0})
        ]
        partition = sc.Partition(mock_bins)
        self.assertEqual(partition.num_bins, 3)

        mock_feature = Feature('feature')
        mock_bins = [
            sc.SetBin(mock_feature, 0.0, {-1.0, 0.0}),
            sc.SetBin(mock_feature, 0.0, {-0.1, 1.0}),
            sc.SetBin(mock_feature, 0.0, {1.0, 2.0})
        ]
        self.assertRaises(ValueError, sc.Partition, mock_bins)

    def test_evaluate(self):
        assignment = {'feature': 1.0}

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=1.0)
        mock_bins = [
            sc.LinearIntervalBin(mock_feature, 0.1, 0.25, -1.0, 0.0),
            sc.LinearIntervalBin(mock_feature, -0.1, -0.25, 0.0, 1.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 1.0, 1.0, 2.0)
        ]
        partition = sc.Partition(mock_bins)
        self.assertEqual(partition.evaluate(assignment), 1.0)

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.5)
        mock_bins = [
            sc.LinearIntervalBin(mock_feature, 0.1, 0.25, -1.0, 0.0),
            sc.LinearIntervalBin(mock_feature, -0.1, -0.25, 0.0, 1.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 1.0, 1.0, 2.0)
        ]
        partition = sc.Partition(mock_bins)
        self.assertEqual(partition.evaluate(assignment), -0.225)

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.0)
        mock_bins = [
            sc.SetBin(mock_feature, 0.1, {-1.0, 0.0}),
            sc.SetBin(mock_feature, -0.1, {0.5, 1.0}),
            sc.SetBin(mock_feature, 0.0, {1.5, 2.0})
        ]
        partition = sc.Partition(mock_bins)
        self.assertEqual(partition.evaluate(assignment), 0.1)

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=1.1)
        mock_bins = [
            sc.LinearIntervalBin(mock_feature, 0.1, 0.25, -1.0, 0.0),
            sc.LinearIntervalBin(mock_feature, -0.1, -0.25, 0.0, 1.0),
            sc.LinearIntervalBin(mock_feature, 0.0, 1.0, 1.5, 2.0)
        ]
        partition = sc.Partition(mock_bins)
        self.assertRaises(ValueError, partition.evaluate, assignment)

        mock_feature = Feature('feature')
        mock_feature.evaluate = MagicMock(return_value=0.1)
        mock_bins = [
            sc.SetBin(mock_feature, 0.1, {-1.0, 0.0}),
            sc.SetBin(mock_feature, -0.1, {0.5, 1.0}),
            sc.SetBin(mock_feature, 0.0, {1.5, 2.0})
        ]
        partition = sc.Partition(mock_bins)
        self.assertRaises(ValueError, partition.evaluate, assignment)
