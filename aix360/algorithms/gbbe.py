import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class GlobalBBExplainer(ABC):
    """
    GlobalBBExplainer is the base class for Global black-box explainers (GBBE).
    Given a source black box model, such explainers generally train a surrogate 
    model that is explainable.

    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize a GlobalBBExplainer object.
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
