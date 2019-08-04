# Directly interpretable supervised explainers

import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class DISExplainer(ABC):
    """
    DISExplainer is the base class for Directly Interpretable Supervised Explainers (DISE).
    Such explainers attempt to train explainable models directly from data.
    Examples include Boolean Decision Rules via Column Generation[#1]_, Generalized Linear Rule Models[#2]_,
    TED: Teaching AI to Explain its Decisions[#3]_, etc.

    References:
        .. [#1] Boolean Decision Rules via Column Generation, NIPS 2018.
        Dash, Gunluk, Wei. https://arxiv.org/abs/1805.09901
        .. [#2] Generalized Linear Rule Models, ICML 2019.
        Dennis Wei, Sanjeeb Dash, Tian Gao, Oktay Gunluk.
        https://arxiv.org/abs/1906.01761
        .. [#3] TED: Teaching AI to Explain its Decisions, AIES 2019.
        Michael Hind, Dennis Wei, Murray Campbell, Noel C. F. Codella, Amit Dhurandhar,
        Aleksandra Mojsilovic, Karthikeyan Natesan Ramamurthy, Kush R. Varshney.
        https://arxiv.org/abs/1811.04896
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize a DISExplainer object.
        ToDo: check common steps that need to be distilled here.
        """

    @abc.abstractmethod
    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def explain(self, *argv, **kwargs):
        """
        Explain the model
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, *argv, **kwargs):
        """
        Fit a model on data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *argv, **kwargs):
        """
        Prediction for a batch of inputs (numpy array)
        """
        raise NotImplementedError
