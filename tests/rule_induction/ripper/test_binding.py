from unittest import TestCase

import numpy as np

from aix360.algorithms.rule_induction.ripper import binding


class TestBinding(TestCase):
    def test_filter_contradicted_instances_happy_path(self):
        pos = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )

        neg = np.array(
            [[7, 8, 9],
             [4, 5, 6]]
        )
        actual = binding._filter_contradicted_instances(pos, neg)
        expected = np.array([[7, 8, 9]])
        np.testing.assert_equal(actual, expected)

    def test_filter_contradicted_instances_same_pos_neg(self):
        pos = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )

        neg = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        actual = binding._filter_contradicted_instances(pos, neg)
        expected = np.array([]).reshape(0, 3)
        np.testing.assert_equal(actual, expected)

    def test_filter_contradicted_instances_disjoint_pos_neg(self):
        pos = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )

        neg = np.array(
            [[7, 8, 9]]
        )
        actual = binding._filter_contradicted_instances(pos, neg)
        expected = np.array([[7, 8, 9]])
        np.testing.assert_equal(actual, expected)
