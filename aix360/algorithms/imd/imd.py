from ..dise import DISExplainer

import numpy as np
import pandas as pd

from aix360.algorithms.imd.utils import _parse_feature_names
from aix360.algorithms.imd.jst import JointSurrogateTree
from aix360.algorithms.imd.rule import Rule


class IMDExplainer(DISExplainer):
    """
    Interpretable Model Differencing to explain the similarities and differences between two classifiers.
    Provides access to :class:`aix360.algorithms.imd.jst.JointSurrogateTree`, a novel data structure to
    compactly represent the differences between the models in terms of rules, and also provides a way to
    visualize the joint surrogate tree structure.

    References:
        .. [#UAI2023] `S. Haldar, D. Saha, D. Wei, R. Nair, E. M. Daly, "Interpretable Differencing of Machine Learning
            Models." Uncertainty in Artificial Intelligence (UAI), 2023.`
    """

    def __init__(self):
        """
        Initialize an IMDExplainer object.
        """

        super(IMDExplainer, self).__init__()

        # to be populated on calling fit() method, or set manually
        self.jst = None
        self.diffrules = []
        self.diffregions = []  # regions in which the model differ, for all feature {'feature': [min, max],...}
        self.feature_names = []
        self.is_int_col = dict()

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def fit(self, X_train: pd.DataFrame, Y1, Y2, max_depth, split_criterion=1, alpha=0.0, verbose=True, **kwargs):
        """
        Fit joint surrogate tree to input data, and outputs from two models.
        Args:
            X_train: input dataframe
            Y1: model1 outputs
            Y2: model2 outputs
            max_depth: maximum depth of the joint surrogate tree to be built
            feature_names: list of input feature names
            alpha: parameter to control degree of favouring common nodes vs. separate nodes
            split_criterion: which divergence criterion to use? (see paper for more details)
            verbose:
            **kwargs:
        Returns:
            self
        """
        feature_names = X_train.columns.to_list()
        self.feature_names = feature_names

        x1 = x2 = X_train.to_numpy()

        if not isinstance(Y1, np.ndarray):
            Y1 = Y1.to_numpy()
        if not isinstance(Y2, np.ndarray):
            Y2 = Y2.to_numpy()

        y1 = Y1
        y2 = Y2

        ydiff = (y1 != y2).astype(int)
        if verbose:
            print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.2f}%")

        jstobj = JointSurrogateTree(max_depth=max_depth,
                                    feature_names=feature_names,
                                    split_criterion=split_criterion,
                                    alpha=alpha)
        t1, t2 = jstobj.fit(x1, y1, x2, y2)
        ct = jstobj.common_trunk(t1, t2)
        diffrules = jstobj.get_diffrules_from_jst(ct)

        self.jst = ct
        self.diffrules = diffrules

        # prepare regions from the rules
        cat_dict, nums = _parse_feature_names(feature_names)
        is_int_col = dict()
        for num_feature in nums:
            is_int_col[num_feature] = np.array_equal(X_train[num_feature], X_train[num_feature].astype(int))
        self.is_int_col = is_int_col

        minimums = X_train.to_numpy().min(0)
        maximums = X_train.to_numpy().max(0)

        fd = [[minimums[i], maximums[i]] for i in range(len(minimums))]
        total_region_dict = {feature_names[i]: fd[i] for i in range(len(feature_names))}

        # some unnecessary wrapping to reuse some code
        diffruledict = dict(enumerate(self.diffrules))
        self.diffregions = [
            rule.as_dict(feature_names=self.feature_names, total_region=total_region_dict, only_preds=False)
            for _, rule in diffruledict.items()
        ]

    def predict(self, X, *argv, **kwargs):
        """Predict diff-labels.
        """
        pass

    def explain(self, *argv, **kwargs):
        """Return diff-rules.
        """
        return self.diffrules

    def inregion(self, regions, data):
        # utility to return a boolean array, True if a datarow is satisfied by one region in `regions`
        # data is numpy array, `regions` is self.diffregions
        res1 = np.zeros(data.shape[0])

        for region in regions:
            res = np.ones(data.shape[0])
            for field in region.keys():
                f_index = self.feature_names.index(field)
                col = data[:, f_index]
                left = region[field][0]
                right = region[field][1]
                res = np.logical_and(res, np.logical_and(col >= left, col <= right))

            res1 = np.logical_or(res1, res)

        return res1

    def metrics(self, x_test: pd.DataFrame, y_test1, y_test2, name="test"):
        """
        take x_test and check the precision and recall
        precision =
           number of actual diff samples inside the diffregion /
           number of test samples inside the diffregion
        recall = diff samples inside the region / total number of diff samples


        Args:
            x_test: test data (only x) to compute diff-based metrics
            name: string (`train` or `test`)

        Returns:
            a dictionary having `precision`, `recall`, `num-rules`, and `num-unique-preds` values
            as obtained from the diff-rules extracted from the jst.
        """

        if self.jst is None:
            print("jst not fitted yet, please call .fit method first!")
            return {}

        metrics = {}
        # y_test1 = self.model1.predict(x_test)
        # y_test2 = self.model2.predict(x_test)
        diff_samples = y_test1 != y_test2
        total_number_diff_samples = np.sum(diff_samples)
        metrics["diffs"] = total_number_diff_samples
        metrics["samples"] = len(x_test)

        inregiondiff = self.inregion(self.diffregions, x_test[diff_samples].to_numpy())
        diff_samples_inside_diff_region = np.sum(inregiondiff)
        inregion = self.inregion(self.diffregions, x_test.to_numpy())
        samples_in_region = np.sum(inregion)

        metrics[name + "-precision"] = round(diff_samples_inside_diff_region / samples_in_region, 6)
        metrics[name + "-recall"] = round(diff_samples_inside_diff_region / total_number_diff_samples, 6)
        metrics["num-rules"] = len(self.diffregions)

        preds = []
        for rule in self.diffrules:
            preds += rule.predicates
        preds = set(preds)
        metrics["num-unique-preds"] = len(preds)
        return metrics

