from collections import namedtuple
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from aix360.algorithms.rule_induction.trxf.core.utils import batch_evaluate

RuleContributionMetrics = namedtuple(
    'RuleContributionMetrics',
    'conjunction, tp_common, tp_unique, fp_common, fp_unique, n_correct, n_records, laplace_estimate')
"""
conjunction: a trxf Conjunction object
tp_common: count of true positives identified based on this and at least one other rule (large values indicate redundancy)
tp_unique: count of true positives correctly identifying a label based only on this rule (indicates strength of the rule)
fp_common: count of false positives based on this and at least one other rule
fp_unique: count of false positives, i.e. mis-classified instances, based on this rule alone 
n_correct: count of true positives + true_negatives
n_records: total number of instances
laplace_estimate: metric of confidence (https://en.wikipedia.org/wiki/Additive_smoothing)
"""

RuleComplexityMetrics = namedtuple('RuleComplexityMetrics',
                                   'num_rules, max_rule_length, sum_rule_length, num_distinct_predicates')


def compute_ruleset_metrics(ruleset: DnfRuleSet, X: pd.DataFrame, y: pd.Series):
    """
    Compute ruleset-level confusion matrix and accuracy when used as a binary classifier

    @param X: features
    @param y: labels
    @return: a dict describing the confusion matrix and accuracy
    """
    then_part = ruleset.then_part
    y_true = (y == then_part)
    y_pred = batch_evaluate(ruleset, X)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tn + tp) / len(y)
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy}


def compute_rule_metrics(ruleset: DnfRuleSet, X: pd.DataFrame, y: pd.Series) -> List[RuleContributionMetrics]:
    """
    Compute individual rule-level contribution to coverage and redundancy.
    See https://ibm.ent.box.com/file/888961054325?sb=/activity/annotations/956283151925 for more context.

    @param X: features
    @param y: labels
    @return: a dict mapping the string representation of conjunctions to a dict containing the contribution metrics
    """
    then_part = ruleset.then_part
    res = []
    for conjunction in ruleset.conjunctions:
        complement = deepcopy(ruleset)
        complement.remove_conjunction(conjunction)
        y_pred_compl = batch_evaluate(complement, X)
        tp_compl, tn_compl, fp_compl, fn_compl = get_preaggregated_confusion_matrix(y_pred_compl, y, then_part)
        y_pred_conj = batch_evaluate(conjunction, X)
        tp_conj, tn_conj, fp_conj, fn_conj = get_preaggregated_confusion_matrix(y_pred_conj, y, then_part)
        laplace_estimate = (len(tp_conj) + 1) / (len(tp_conj) + len(fp_conj) + 2)

        tp_common = len(set(tp_conj).intersection(set(tp_compl)))
        tp_unique = len(set(tp_conj).difference(set(tp_compl)))
        fp_common = len(set(fp_conj).intersection(set(fp_compl)))
        fp_unique = len(set(fp_conj).difference(set(fp_compl)))
        n_correct = len(tn_conj) + len(tp_conj)
        n_records = X.shape[0]
        res.append(RuleContributionMetrics(
            conjunction, tp_common, tp_unique, fp_common, fp_unique, n_correct, n_records, laplace_estimate))
    return res


def compute_ruleset_complexity(ruleset: DnfRuleSet):
    num_rules = len(ruleset)
    rule_lengths = list(map(len, ruleset.conjunctions))
    max_rule_length = max(rule_lengths)
    sum_rule_length = sum(rule_lengths)
    predicates = []
    for c in ruleset.conjunctions:
        predicates += [p for p in c.predicates if p not in predicates]
    num_distinct_predicates = len(predicates)
    return RuleComplexityMetrics(num_rules, max_rule_length, sum_rule_length, num_distinct_predicates)


def get_preaggregated_confusion_matrix(y_pred, y_true, pos_value):
    y_p = y_pred.to_numpy()
    y_t = (y_true.to_numpy() == pos_value)

    tp = np.where(y_p & y_t)[0]
    tn = np.where(np.logical_not(y_p) & np.logical_not(y_t))[0]
    fp = np.where(y_p & np.logical_not(y_t))[0]
    fn = np.where(y_t & np.logical_not(y_p))[0]

    return tp, tn, fp, fn
