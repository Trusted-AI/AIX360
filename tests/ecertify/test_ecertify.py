import unittest
import numpy as np

from aix360.algorithms.ecertify.ExpCertifyBB import Certify

class TestCertifyExplainer(unittest.TestCase):

    def test_all_imports(self):
        from aix360.algorithms.ecertify.ExpCertifyBB import Certify, Ecertify
        from aix360.algorithms.ecertify.utils import compute_lime_explainer, compute_shap_explainer

    def test_for_low_dim_toy_problem(self):
        #Calling Function
        d = 2 #input dimensionality
        x = np.array([0.0]*d) #input
        theta = 0.75 #fidelity threshold
        Z = 10 #number of hypercubes to certify
        Q = 1000 #query budget for each hypercube
        eps = 0.1/d #min gap between lb and ub
        numruns = 10

        
        pass

    def test_for_fico(self):
        # just run
        pass


if __name__ == '__main__':
    unittest.main()