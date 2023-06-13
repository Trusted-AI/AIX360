from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Rule:
    """
        a class to represent a single clause/rule of the form:
        "IF <literal1> AND <literal2> AND ... <literal_k> THEN predict class = 0 or 1"
    """
    _id: int
    predicates: list
    class_label: int

    def as_string(self):
        # todo: add check for null rule (not necessary)
        s = "IF"
        for idx, pred in enumerate(self.predicates):
            f, op, th = pred
            s += " {} {} {:.4f}".format(f, op, th)
            if idx != len(self.predicates) - 1:
                s += " AND"
        s += " THEN PREDICT class = {}".format(self.class_label)
        return s

    def check_equal_preds(self, rule2):
        preds1 = self.predicates
        preds2 = rule2.predicates

        if len(preds1) != len(preds2):
            return False
        for pred1 in preds1:
            if pred1 not in preds2:
                return False

        return True

    def as_dict(self, feature_names: list=None, total_region: dict=None, only_preds=False):
        """
        returns the dict-like representation of the rule, having ranges for all features, if arguments are passed.
        useful for computing intersection, and generating data in the region.
        Args:
            feature_names: list of all feature names in the training data
            total_region: range observed for each feature in the training data
                            if not passed, it is taken as [-inf, inf]
            only_preds: do you want a dictionary representation with only the features/predicates present
                        in this rule?
                        if only_preds is True, the returned dictionary contains features that appear in self.predicates,
                        only useful if you also pass all `feature_names` and `total_region`.
        Returns:
            region: a dictionary whose keys are feature names, and values are a 2 element list (min-val, max-val
                    permissible for the feature in that rule)
        """

        if feature_names is None:  # then build dict with only appearing features
            feature_names = []
            for pred in self.predicates:
                f, op, th = pred
                feature_names.append(f)
        feature_names = list(set(feature_names))

        region = {}

        for feature in feature_names:
            if total_region is not None:
                minvalue = total_region[feature][0]
                maxvalue = total_region[feature][1]
            else:  # if ranges not passed, use [-inf, inf] range.
                minvalue = np.NINF
                maxvalue = np.Inf

            predicates_having_feature = []

            for pred in self.predicates:
                f, op, th = pred
                if f == feature:
                    predicates_having_feature.append(pred)

            for idx, pred in enumerate(predicates_having_feature):
                f, op, th = pred
                if op == '<=' or op == '<':
                    if th < maxvalue:
                        maxvalue = th
                    pass
                elif op == '>' or op == '>=':
                    if th > minvalue:
                        minvalue = th

            if not only_preds:
                region[feature] = [minvalue, maxvalue]
            elif len(predicates_having_feature) != 0:
                region[feature] = [minvalue, maxvalue]

        return region

    def __repr__(self):
        return self.as_string()

    def apply(self, X: pd.DataFrame):
        """
        Args:
            X: pandas dataframe

        Returns:
            a boolean array `res` of length `X.shape[0]` (no. of instances in X)
            `res[i] = True => this rule satisfies X[i]`
        """
        res = np.ones(X.shape[0], dtype='bool')  # index array
        for idx, pred in enumerate(self.predicates):
            f, op, th = pred
            if op == '<=' or op == '<':
                if idx == 0:
                    res = (X[f] <= th).to_numpy()
                else:
                    res = res * (X[f] <= th).to_numpy()
            elif op == '>' or op == '>=':
                if idx == 0:
                    res = (X[f] >= th).to_numpy()
                else:
                    res = res * (X[f] >= th).to_numpy()
            else:
                raise ValueError('op must be <= or >')
        return res

    def filter(self, X: pd.DataFrame):
        return X[self.apply(X)]  # filters the rows where self.apply(X) is true.

    def predict(self, X: pd.DataFrame=None):
        raise NotImplementedError

    @staticmethod
    def intersect_dicts(region1: dict, region2: dict):
        result = {}
        for k in region1.keys():
            if k in region2:
                l1 = region1[k][0]
                l2 = region1[k][1]
                r1 = region2[k][0]
                r2 = region2[k][1]
                bin = (l2 == r1) or (r2 == l1)
                if r2 < l1 or r1 > l2 or bin:  # = because what if one is [0, 0.5] and other is [0.5, 1] ?
                    # intersection region is NULL set, no value is admissible for this feature
                    return {}
                else:
                    result[k] = [max(r1, l1), min(l2, r2)]
            else:
                result[k] = [region1[k][0], region1[k][1]]
        for k in region2.keys():
            if k not in region1:
                result[k] = [region2[k][0], region2[k][1]]

        return result

    def intersection(self, another_leaf):
        """
        Args:
            another_leaf: another rule object

        Returns:
            returns the intersection region of two rules.
                common predicates => take intersected region
                for others (present in one rule exclusively), add as it is from total_region.

            currently class label is not used in calculation.
        """
        region1 = self.as_dict()
        region2 = another_leaf.as_dict()

        region = Rule.intersect_dicts(region1, region2)
        new_preds = []

        new_class_label = int(self.class_label != another_leaf.class_label)  # todo: 1 if they mismatch (redundant)

        for key in region:
            left = region[key][0]
            right = region[key][1]

            if not np.isinf(left):
                new_preds.append((key, '>', left))
            if not np.isinf(right):
                new_preds.append((key, '<=', right))

        new_id = str(self._id) + str(another_leaf._id)
        return Rule(new_id, new_preds, new_class_label)
