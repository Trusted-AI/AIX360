from unittest import TestCase
from unittest.mock import MagicMock
import aix360.algorithms.rule_induction.trxf.scorecard as sc
from aix360.algorithms.rule_induction.trxf.core import Feature


class TestScorecard(TestCase):

    def test_empty_partition(self):
        self.assertRaises(ValueError, sc.Scorecard, [])

    def test_repeated_features(self):
        mock_feature = Feature('feature')
        partitions = [
            sc.Partition([
                sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
            ])
        ]
        scorecard = sc.Scorecard(partitions)
        self.assertEqual(scorecard.num_features, 1)

        mock_feature1 = Feature('feature1')
        mock_feature2 = Feature('feature2')
        partitions = [
            sc.Partition([
                sc.LinearIntervalBin(mock_feature1, 0.0, 0.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature1, 0.0, 0.0, 0.0, 1.0)
            ]),
            sc.Partition([
                sc.SetBin(mock_feature2, 0.0, {-1.0, 0.0}),
                sc.SetBin(mock_feature2, 0.0, {1.0, 2.0})
            ])
        ]
        scorecard = sc.Scorecard(partitions)
        self.assertEqual(scorecard.num_features, 2)

        mock_feature = Feature('feature')
        partitions = [
            sc.Partition([
                sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
            ]),
            sc.Partition([
                sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 1.0, 2.0)
            ])
        ]
        self.assertRaises(ValueError, sc.Scorecard, partitions)

        mock_feature = Feature('feature')
        partitions = [
            sc.Partition([
                sc.LinearIntervalBin(mock_feature, 0.0, 0.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature, 0.0, 0.0, 0.0, 1.0)
            ]),
            sc.Partition([
                sc.SetBin(mock_feature, 0.0, {-1.0, 0.0}),
                sc.SetBin(mock_feature, 0.0, {1.0, 2.0})
            ])
        ]
        self.assertRaises(ValueError, sc.Scorecard, partitions)

        mock_feature1 = Feature('feature1')
        mock_feature2 = Feature('feature2')
        partitions = [
            sc.Partition([
                sc.LinearIntervalBin(mock_feature1, 0.0, 0.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature1, 0.0, 0.0, 0.0, 1.0)
            ]),
            sc.Partition([
                sc.LinearIntervalBin(mock_feature2, 0.0, 0.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature2, 0.0, 0.0, 1.0, 2.0)
            ]),
            sc.Partition([
                sc.LinearIntervalBin(mock_feature1, 0.0, 0.0, -2.0, -1.0),
                sc.LinearIntervalBin(mock_feature1, 0.0, 0.0, 5.0, 10.0)
            ])
        ]
        self.assertRaises(ValueError, sc.Scorecard, partitions)

    def test_evaluate(self):
        assignment = {'feature1': 0.5, 'feature2': 0.0}

        mock_feature1 = Feature('feature1')
        mock_feature2 = Feature('feature2')
        mock_feature1.evaluate = MagicMock(return_value=assignment['feature1'])
        mock_feature2.evaluate = MagicMock(return_value=assignment['feature2'])
        partitions = [
            sc.Partition([
                sc.LinearIntervalBin(mock_feature1, 0.5, -2.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature1, -1.0, 2.0, 0.0, 1.0)
            ]),
            sc.Partition([
                sc.SetBin(mock_feature2, 1.0, {-1.0, 0.0}),
                sc.SetBin(mock_feature2, -1.0, {1.0, 2.0})
            ])
        ]
        scorecard = sc.Scorecard(partitions)
        self.assertEqual(scorecard.evaluate(assignment), 1.0)

        mock_feature1 = Feature('feature1')
        mock_feature2 = Feature('feature2')
        mock_feature1.evaluate = MagicMock(return_value=assignment['feature1'])
        mock_feature2.evaluate = MagicMock(return_value=assignment['feature2'])
        partitions = [
            sc.Partition([
                sc.LinearIntervalBin(mock_feature1, 0.5, -2.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature1, -1.0, 2.0, 0.0, 1.0)
            ]),
            sc.Partition([
                sc.SetBin(mock_feature2, 1.0, {-1.0, 0.0}),
                sc.SetBin(mock_feature2, -1.0, {1.0, 2.0})
            ])
        ]
        scorecard = sc.Scorecard(partitions, bias=-10)
        self.assertEqual(scorecard.evaluate(assignment), -9.0)

    def test_total_bins(self):
        mock_feature1 = Feature('feature1')
        mock_feature2 = Feature('feature2')
        partitions = [
            sc.Partition([
                sc.LinearIntervalBin(mock_feature1, 0.5, -2.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature1, -1.0, 2.0, 0.0, 1.0),
                sc.LinearIntervalBin(mock_feature1, -1.0, 1.0, 1.0, 2.0)
            ]),
            sc.Partition([
                sc.SetBin(mock_feature2, 1.0, {-1.0, 0.0}),
                sc.SetBin(mock_feature2, -1.0, {1.0, 2.0})
            ])
        ]
        scorecard = sc.Scorecard(partitions)
        self.assertEqual(scorecard.total_bins, 5)

    def test_bins_per_feature(self):
        mock_feature1 = Feature('feature1')
        mock_feature2 = Feature('feature2')
        partitions = [
            sc.Partition([
                sc.LinearIntervalBin(mock_feature1, 0.5, -2.0, -1.0, 0.0),
                sc.LinearIntervalBin(mock_feature1, -1.0, 2.0, 0.0, 1.0),
                sc.LinearIntervalBin(mock_feature1, -1.0, 1.0, 1.0, 2.0)
            ]),
            sc.Partition([
                sc.SetBin(mock_feature2, 1.0, {-1.0, 0.0}),
                sc.SetBin(mock_feature2, -1.0, {1.0, 2.0})
            ])
        ]
        scorecard = sc.Scorecard(partitions)
        self.assertListEqual(scorecard.bins_per_feature, [(mock_feature1, 3), (mock_feature2, 2)])
