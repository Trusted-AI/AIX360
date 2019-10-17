from __future__ import print_function

from lime import lime_image, lime_text, lime_tabular
from aix360.algorithms.lbbe import LocalBBExplainer


class LimeTextExplainer(LocalBBExplainer):
    """
    This class wraps the source class `LimeTextExplainer <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_text>`_
    available in the `LIME <https://github.com/marcotcr/lime>`_ library.
    Additional variables or functions from the source class can also be accessed via the 'explainer'
    object variable that is initialized in '__init__' function of this class.
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize lime text explainer object.
        """
        super(LimeTextExplainer, self).__init__(*argv, **kwargs)

        self.explainer = lime_text.LimeTextExplainer(*argv, **kwargs)

    def set_params(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explain one or more input instances.
        """
        self.explanation = self.explainer.explain_instance(*argv, **kwargs)

        return (self.explanation)


class LimeImageExplainer(LocalBBExplainer):
    """
    This class wraps the source class `LimeImageExplainer <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image>`_
    available in the `LIME <https://github.com/marcotcr/lime>`_ library.
    Additional variables or functions from the source class can also be accessed via the 'explainer'
    object variable that is initialized in '__init__' function of this class.
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize lime Image explainer object
        """
        super(LimeImageExplainer, self).__init__(*argv, **kwargs)

        self.explainer = lime_image.LimeImageExplainer(*argv, **kwargs)

    def set_params(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explain one or more input instances.
        """
        self.explanation = self.explainer.explain_instance(*argv, **kwargs)

        return (self.explanation)


class LimeTabularExplainer(LocalBBExplainer):
    """
    This class wraps the source class `LimeTabularExplainer <https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_tabular>`_
    available in the `LIME <https://github.com/marcotcr/lime>`_ library.
    Additional variables or functions from the source class can also be accessed via the 'explainer'
    object variable that is initialized in '__init__' function of this class.
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize lime Tabular Explainer object
        """
        super(LimeTabularExplainer, self).__init__(*argv, **kwargs)

        self.explainer = lime_tabular.LimeTabularExplainer(*argv, **kwargs)

    def set_params(self, verbose=0):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explain one or more input instances.
        """
        self.explanation = self.explainer.explain_instance(*argv, **kwargs)

        return (self.explanation)