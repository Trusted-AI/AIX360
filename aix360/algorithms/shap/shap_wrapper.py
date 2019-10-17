from __future__ import print_function

import shap
from aix360.algorithms.lbbe import LocalBBExplainer
from aix360.algorithms.lwbe import LocalWBExplainer


class KernelExplainer(LocalBBExplainer):
    """
    This class wraps shap.KernelExplainer [#]_.
    For accessing any variables or functions not exposed by this class, please
    access them via the 'explainer' object variable initialized in the '__init__' function.

    References:
        .. [#] `<https://shap.readthedocs.io/>`_
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize shap kernelexplainer object.
        """
        super(KernelExplainer, self).__init__(*argv, **kwargs)

        self.explainer = shap.KernelExplainer(*argv, **kwargs)

    def set_params(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explains an input instance x.
        """
        return (self.explainer.shap_values(*argv, **kwargs))


class GradientExplainer(LocalWBExplainer):
    """
    This class that wraps shap.GradientExplainer [#]_.
    For accessing any variables or functions not exposed by this class, please
    access them via the 'explainer' object variable initialized in the '__init__' function.

    References:
        .. [#] `<https://shap.readthedocs.io/>`_
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize shap kernelexplainer object.
        """
        super(GradientExplainer, self).__init__(*argv, **kwargs)

        self.explainer = shap.GradientExplainer(*argv, **kwargs)

    def set_params(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explains an input instance x.
        """
        return (self.explainer.shap_values(*argv, **kwargs))



class DeepExplainer(LocalWBExplainer):
    """
    This class that wraps shap.DeepExplainer [#]_.
    For accessing any variables or functions not exposed by this class, please
    access them via the 'explainer' object variable initialized in the '__init__' function.

    References:
        .. [#] `<https://shap.readthedocs.io/>`_
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize shap kernelexplainer object.
        """
        super(DeepExplainer, self).__init__(*argv, **kwargs)

        self.explainer = shap.DeepExplainer(*argv, **kwargs)

    def set_params(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explains an input instance x.
        """
        return (self.explainer.shap_values(*argv, **kwargs))



class TreeExplainer(LocalWBExplainer):
    """
    Class that wraps shap.TreeExplainer [#]_.
    For accessing any variables/functions not exposed by this class, please
    access them via the 'explainer' object variable initialized in '__init__' function.

    References:
        .. [#] `<https://shap.readthedocs.io/>`_
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize shap kernelexplainer object.
        """
        super(TreeExplainer, self).__init__(*argv, **kwargs)

        self.explainer = shap.TreeExplainer(*argv, **kwargs)

    def set_params(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explains an input instance x.
        """
        return (self.explainer.shap_values(*argv, **kwargs))



class LinearExplainer(LocalWBExplainer):
    """
    Class that wraps shap.Linearexplainer [#]_.
    For accessing any variables/functions not exposed by this class, please
    access them via the 'explainer' object variable initialized in '__init__' function.

    References:
        .. [#] `<https://shap.readthedocs.io/>`_
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize shap kernelexplainer object.
        """
        super(LinearExplainer, self).__init__(*argv, **kwargs)

        self.explainer = shap.LinearExplainer(*argv, **kwargs)

    def set_params(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explains an input instance x.
        """
        return (self.explainer.shap_values(*argv, **kwargs))

