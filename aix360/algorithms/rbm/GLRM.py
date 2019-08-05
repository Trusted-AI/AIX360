from ..dise import DISExplainer


class GLRMExplainer(DISExplainer):
    """
    Generalized Linear Rule Model explainer. Provides access to the following
    directly interpretable supervised learning methods:

    * **Linear Rule Regression:** linear regression on rule-based features [#Icml2019]_.
    * **Logistic Rule Regression:** logistic regression on rule-based features [#Icml2019]_.

    References:
        .. [#Icml2019] `D. Wei, S. Dash, T. Gao, O. Günlük, "Generalized linear rule
           models." International Conference on Machine Learning (ICML), 2019.
           <http://proceedings.mlr.press/v97/wei19a/wei19a.pdf>`_
    """

    def __init__(self, model):
        """
        Initialize a GLRMExplainer object.

        Args:
          model: model to operate on. Instance of either

            * :class:`aix360.algorithms.rbm.linear_regression.LinearRuleRegression` or
            * :class:`aix360.algorithms.rbm.logistic_regression.LogisticRuleRegression`
        """
        super(GLRMExplainer, self).__init__()
        self._model = model

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def fit(self, X_train, Y_train, Xstd=None):
        """Fit model to training data.

        Args:
            X_train (DataFrame): Binarized features with MultiIndex column labels
            Y_train (array): Target variable
            Xstd (DataFrame, optional): Standardized numerical features
        Returns:
            GLRMExplainer: Self
        """
        self._model.fit(X_train, Y_train, Xstd)

    def predict(self, X, Xstd=None):
        """Predict responses.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
            Xstd (DataFrame, optional): Standardized numerical features
        Returns:
            array: y -- Predicted responses
        """
        return self._model.predict(X, Xstd)

    def predict_proba(self, X, Xstd=None):
        """Predict probabilities of Y=1. Only available if underlying model implements
        `predict_proba` method.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
            Xstd (DataFrame, optional): Standardized numerical features
        Returns:
            array: p -- Predicted probabilities

        Raises:
            ValueError: if model doesn't implement `predict_proba`
        """
        if hasattr(self._model, 'predict_proba'):
            return self._model.predict_proba(X, Xstd)
        raise ValueError("Current model doesn't predict probabilities")

    def explain(self, maxCoeffs=None, highDegOnly=False, prec=2):
        """Return DataFrame holding model features and their coefficients.

        Args:
            maxCoeffs (int, optional): Maximum number of rules/numerical features to show
            highDegOnly (bool, optional): Only show higher-degree rules
            prec (int, optional): Number of decimal places to show for floating-value thresholds
        Returns:
            DataFrame: dfExpl -- Rules/numerical features and their coefficients
        """
        return self._model.explain(maxCoeffs, highDegOnly, prec)

    def visualize(self, Xorig, fb, features=None):
        """Plot generalized additive model component, which includes first-degree rules 
        and linear functions of unbinarized ordinal features but excludes higher-degree rules.

        Args:
            Xorig (DataFrame): Original unbinarized features
            fb: FeatureBinarizer object used to binarize features
            features (list, optional): Subset of features to be plotted
        """
        return self._model.visualize(Xorig, fb, features)
