import unittest
import os
import json

from typing import Dict



# helper for testing an optimal transport case
def _test_ot(
    tester,
    data: Dict,
):
    import numpy as np

    from aix360.algorithms.matching import OTMatchingExplainer
    explainer = OTMatchingExplainer()

    v = explainer.explain_instance(
        matching=np.array(data['matching']),
        costs=np.array(data['costs']),
        constraints=(
            np.array(data['constraint_a']),
            np.array(data['constraint_b'])
        ),
        num_alternate_matchings=data['num_alternate_matchings'],
        search_node_limit=data['search_node_limit'],
        search_depth_limit=data['search_depth_limit'],
        search_match_pos_filter=[
            tuple(x) for x in data['search_match_pos_filter']
        ],
    )

    for i, (o, d) in enumerate(zip(v, data['explanations'])):
        with tester.subTest(instance=i):

            tester.assertEqual(
                o.salient, [tuple(x) for x in d['salient']],
                'wrong salient positions',
            )

            tester.assertAlmostEqual(
                np.multiply(
                    o.matching, 
                    np.array(data['costs']),
                ).sum(),
                d['cost'], 
                places=3, msg="cost not equal",
            )

class TestOTMatchingExplainer(unittest.TestCase):

    def test_ot_matcher(self):
        _dirpath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data'
        )

        for case in [
            'test_case_1',
        ]:
            with open(
                os.path.join(_dirpath, case + '.json'), 'r'
            ) as f:
                CASE = json.load(f)

            with self.subTest(instance=case):
                _test_ot(self, CASE)

if __name__ == '__main__':
    unittest.main()
