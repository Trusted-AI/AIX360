import unittest
import numpy as np

from aix360.algorithms.ecertify.ExpCertifyBB import Ecertify
from aix360.algorithms.ecertify.ecertify import CertifyExplanation

class TestCertifyExplainer(unittest.TestCase):

    def test_all_imports(self):
        from aix360.algorithms.ecertify.ecertify import CertifyExplanation
        from aix360.algorithms.ecertify.ExpCertifyBB import Certify, Ecertify
        from aix360.algorithms.ecertify.utils import compute_lime_explainer, compute_shap_explainer

    def test_for_low_dim_toy_problem(self):
        def bb(x): #Black box function
            alpha = np.ones((len(x)))
            return alpha.dot(x)

        def e(x): #Explanation function
            beta = 0.75*np.ones((len(x)))
            return beta.dot(x)

        def f(x): #Fidelity function
            #fidelity = 1-abs(bb(x) - e(x))/max(abs(bb(x)), abs(e(x))) #Normalized MAE
            fidelity = 1-abs(bb(x) - e(x)) #MAE
            return fidelity
        
        d=2
        x = np.array([0.0]*d)
        s = 3
        certifier = CertifyExplanation(theta=0.75, Q=1000, Z=10, lb=0, ub=1, sigma0=0.1, numruns=100)
        w = certifier.certify_instance(instance=x, quality_criterion=f, strategy=s, choice='max', silent=False)
        pass

    def test_for_fico(self):
        # just run
        pass


if __name__ == '__main__':
    unittest.main()