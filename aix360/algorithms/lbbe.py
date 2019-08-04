import abc
import sys

# Ensure compatibility with Python 2/3 
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class LocalBBExplainer(ABC):
        
    """ 
    LocalBBExplainer is the base class for local post-hoc black-box explainers (LBBE).
    Such explainers are model agnostic and generally require access to model's predict function alone.
    Examples include LIME[#1]_, SHAP[#2]_, etc..

    References:
        .. [#1] “Why Should I Trust You?” Explaining the Predictions of Any Classifier, ACM SIGKDD 2016.
        Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. https://arxiv.org/abs/1602.04938.
        .. [#2] A Unified Approach to Interpreting Model Predictions, NIPS 2017.
        Lundberg, Scott M and Lee, Su-In. https://arxiv.org/abs/1705.07874

    """ 
    def __init__(self, *argv, **kwargs):
        
        """
        Initialize a LocalBBExplainer object. 
        ToDo: check common steps that need to be distilled here. 
        """
                
    @abc.abstractmethod 
    def set_params(self, *argv, **kwargs): 
        """
        Set parameters for the explainer. 
        """
        raise NotImplementedError 
        
    
    @abc.abstractmethod 
    def explain_instance(self, *argv, **kwargs):
        """
        Explain an input instance x.
        """
        raise NotImplementedError 
