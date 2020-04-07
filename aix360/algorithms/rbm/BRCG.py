from ..dise import DISExplainer


class BRCGExplainer(DISExplainer):
    """
    Boolean Rule Column Generation explainer. Provides access to 
    :class:`aix360.algorithms.rbm.boolean_rule_cg.BooleanRuleCG`, which
    implements a directly interpretable supervised learning method
    for binary classification that learns a Boolean rule in disjunctive
    normal form (DNF) or conjunctive normal form (CNF) using column generation (CG).
    AIX360 implements a heuristic beam search version of BRCG that is less 
    computationally intensive than the published integer programming version [#NeurIPS2018]_.

    References:
        .. [#NeurIPS2018] `S. Dash, O. Günlük, D. Wei, "Boolean decision rules via
           column generation." Neural Information Processing Systems (NeurIPS), 2018.
           <https://papers.nips.cc/paper/7716-boolean-decision-rules-via-column-generation.pdf>`_
    """

    def __init__(self, model):
        """
        Initialize a BRCGExplainer object.

        Args:
          model: model to operate on, instance of :class:`aix360.algorithms.rbm.boolean_rule_cg.BooleanRuleCG`
        """
        super(BRCGExplainer, self).__init__()
        self._model = model

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def fit(self, X_train, Y_train, *argv, **kwargs):
        """Fit model to training data.

        Args:
            X_train (DataFrame): Binarized features with MultiIndex column labels
            Y_train (array): Binary-valued target variable
        Returns:
            BRCGExplainer: Self
        """
        self._model.fit(X_train, Y_train, **kwargs)

    def predict(self, X, *argv, **kwargs):
        """Predict class labels.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
        Returns:
            array: y -- Predicted labels
        """
        return self._model.predict(X, **kwargs)

    def explain(self, *argv, **kwargs):
        """Return rules comprising the underlying model.

        Args:
            maxConj (int, optional): Maximum number of conjunctions to show
            prec (int, optional): Number of decimal places to show for floating-value thresholds
        Returns:
            Dictionary containing

            * isCNF (bool): flag signaling whether model is CNF or DNF
            * rules (list): selected conjunctions formatted as strings
        """
        return self._model.explain(**kwargs)
