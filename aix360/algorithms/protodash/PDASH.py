from __future__ import print_function

from aix360.algorithms.die import DIExplainer

from .PDASH_utils import HeuristicSetSelection


class ProtodashExplainer(DIExplainer):
    """
    ProtodashExplainer provides exemplar-based explanations for summarizing datasets as well
    as explaining predictions made by an AI model. It employs a fast gradient based algorithm
    to find prototypes along with their (non-negative) importance weights. The algorithm minimizes the maximum
    mean discrepancy metric and has constant factor approximation guarantees for this weakly submodular function. [#]_.

    References:
        .. [#] `Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi,
           "ProtoDash: Fast Interpretable Prototype Selection"
           <https://arxiv.org/abs/1707.01212>`_
    """

    def __init__(self):
        """
        Constructor method, initializes the explainer
        """
        super(ProtodashExplainer, self).__init__()

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def explain(self, X, Y, m, kernelType='other', sigma=2, optimizer='cvxpy'):
        """
        Return prototypes for data X, Y.

        Args:
            X (double 2d array): Dataset you want to explain.
            Y (double 2d array): Dataset to select prototypical explanations from.
            m (int): Number of prototypes
            kernelType (str): Type of kernel (viz. 'Gaussian', / 'other')
            sigma (double): width of kernel
            optimizer (string): qpsolver ('cvxpy' or 'osqp')
            
        Returns:
            m selected prototypes from X and their (unnormalized) importance weights
        """
        return( HeuristicSetSelection(X, Y, m, kernelType, sigma, optimizer) )
