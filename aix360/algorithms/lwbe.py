# Module for local white box explainer base classes for text, image, and tabular data. 
# All local WB explainer algorithms (e.g. CEM, etc.) would inherit these classes. 

import abc 
import sys

# Ensure compatibility with Python 2/3 
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class LocalWBExplainer(ABC):
        
    """ 
    LocalWBExplainer is the base class for local post-hoc white box explainers (LBBE).
    Such explainers generally require access to model's internals beyond its predict function.
    Examples include Contrastive explanation method[#1]_, Layer-wise Relevance Propagation[#2]_, etc.

    References:
        .. [#] Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives,
           NIPS, 2018. Amit Dhurandhar, Pin-Yu Chen, Ronny Luss, Chun-Chen Tu,
           Paishun Ting, Karthikeyan Shanmugam, Payel Das. https://arxiv.org/abs/1802.07623
        .. [#2] http://www.heatmapping.org/

    """ 
    def __init__(self, *argv, **kwargs):
        
        """
        Constructor method, initialize a LocalBBExplainer object.
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
        


