import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class GlobalWBExplainer(ABC):
    """
    GlobalWBExplainer is the base class for Global white-box explainers (DIE).
    Given a source model, such explainers generally train a surrogate that is explainable.
    Examples include ProfWt[#1], etc.

    References:
        .. [#1] Improving Simple Models with Confidence Profiles (ProfWt),
        NIPS 2018, Amit Dhurandhar, Karthikeyan Shanmugam, Ronny Luss, Peder Olsen.
        https://arxiv.org/abs/1807.07506
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize a GlobalWBExplainer object.
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
        Explain model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, *argv, **kwargs):
        """
        Train a surrogate model.
        """
        raise NotImplementedError
