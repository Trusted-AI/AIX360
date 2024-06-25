from ..lbbe import LocalBBExplainer

import timeit
import numpy as np
import pandas as pd

from aix360.algorithms.ecertify.ExpCertifyBB import Ecertify

class CertifyExplanation(LocalBBExplainer):
    """
    Explanation certification class for black-box probabilistic certification of local explanations.
    Provides access to :class:`aix360.algorithms.ecertify.ExpCertifyBB`, where the certification strategies
    are implemented.

    References:
        .. [#ICML2024] `A. Dhurandhar, S. Haldar, D. Wei, K. N. Ramamurthy, "Trust Regions for Explanations via
        Black-Box Probabilistic Certification." Forty-first International Conference on Machine Learning (ICML), 2024.`
    """

    def __init__(self, theta, Q,
                 Z=10,
                 lb=0, ub=1,
                 sigma0=0.1,
                 numruns=100, *argv, **kwargs):
        super(CertifyExplanation).__init__(*argv, **kwargs)
        # initialize an ecertify object
        self.theta = theta
        self.Q = Q

        self.Z = Z
        self.lb=lb 
        self.ub=ub
        self.sigma0=sigma0
        self.numruns=numruns

    def set_params(self, *argv, **kwargs): 
        """
        Set parameters for the explainer. 
        """
        raise NotImplementedError 
        
    
    def explain_instance(self, *argv, **kwargs):
        """
        Explain an input instance x.
        """
        raise NotImplementedError
    
    def certify_instance(self, instance, quality_criterion,
                         strategy=3,
                         choice="min",
                         silent=True
                         ):
        """ invokes the Ecertify algorithm to do the certification

        Args:
            instance (_type_): x0
            quality_criterion (_type_): quality criterion
            strategy (int, optional): certification strategies. Defaults to 3 (adaptI).
            choice (str, optional): _description_. Defaults to "min".
        Returns:
            mean certified halfwidth
        """
        certicubeperrun = np.zeros(self.numruns)
        t_0 = timeit.default_timer()
        for irun in range(self.numruns):
            Certicube, _ = Ecertify(instance, self.theta, self.Z, self.Q, self.lb, self.ub, self.sigma0, strategy, quality_criterion, choice=choice)
            certicubeperrun[irun] = Certicube
        t_1 = timeit.default_timer()
        time_per_run = round((t_1 - t_0) / self.numruns, 3)

        if not silent:
            print(f"Time per run: {time_per_run} s")
            print(f"Found w: {np.mean(certicubeperrun):.4f} \u00B1 {np.std(certicubeperrun)/np.sqrt(self.numruns):.6f}")
        
        return certicubeperrun.mean()

