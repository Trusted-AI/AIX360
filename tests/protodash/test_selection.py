import itertools
import unittest

import numpy as np

from aix360.algorithms.protodash import ProtodashExplainer


class TestPrototypeSelection(unittest.TestCase):
    def test_first_prototype_maximizes_gaussian_objective(self):
        for candidates in ([[0.0], [10.0]], [[10.0], [0.0]]):
            with self.subTest(candidates=candidates):
                weights, indices, values = ProtodashExplainer().explain(
                    np.array([[0.0]]), np.array(candidates), 1,
                    kernelType='Gaussian', sigma=1,
                )
                self.assertEqual(candidates[indices[0]], [0.0])
                np.testing.assert_allclose(weights, [1.0])
                np.testing.assert_allclose(values, [0.5])

    def test_first_prototype_maximizes_linear_objective(self):
        candidates = np.array([[1.0, 0.0], [0.0, 1.0]])
        weights, indices, values = ProtodashExplainer().explain(
            np.array([[1.0, 0.0]]), candidates, 1,
        )
        self.assertEqual(indices.tolist(), [0])
        np.testing.assert_allclose(weights, [1.0])
        np.testing.assert_allclose(values, [0.5])

    def test_later_prototype_maximizes_gradient(self):
        for optimizer in ('cvxpy', 'osqp'):
            for order in itertools.permutations([0.0, 1.0, 10.0]):
                with self.subTest(optimizer=optimizer, order=order):
                    candidates = np.array(order).reshape(-1, 1)
                    _, indices, values = ProtodashExplainer().explain(
                        np.array([[0.0], [1.0]]), candidates, 2,
                        kernelType='Gaussian', sigma=1, optimizer=optimizer,
                    )
                    self.assertEqual(set(candidates[indices, 0]), {0.0, 1.0})
                    self.assertGreater(values[1], values[0])


if __name__ == '__main__':
    unittest.main()
