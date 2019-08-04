import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class DIExplainer(ABC):
    """
    DIExplainer is the base class for Directly Interpretable unsupervised explainers (DIE).
    Such explainers generally rely on unsupervised techniques to explain datasets and model predictions.
    Examples include DIP-VAE[#1]_, Protodash[#2]_, etc.

    References:
        .. [#1] Variational Inference of Disentangled Latent Concepts from Unlabeled Observations (DIP-VAE), ICLR 2018.
         Kumar, Sattigeri, Balakrishnan. https://arxiv.org/abs/1711.00848
        .. [#2] ProtoDash: Fast Interpretable Prototype Selection, 2017.
        Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi.
        https://arxiv.org/abs/1707.01212
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize a DIExplainer object.
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
        Explain the data or model.
        """
        raise NotImplementedError
