import numba as nb
import numpy as np

from aix360.algorithms.rule_induction.ripper.binding import _rule_predict, _rule_list_predict
from aix360.algorithms.rule_induction.ripper.mdl import _mdl


@nb.njit(
    'double(int64, int64, int64, int64)',
    nogil=True,
    cache=True,
    error_model='numpy'
)
def accuracy_tp_fp(
        p0,
        n0,
        tp,
        fp
):
    """
    Calculate accuracy using true positive and false positive.
    """
    return (tp + n0 - float(fp)) / (p0 + float(n0))


@nb.njit(
    'double(int64, int64)',
    nogil=True,
    cache=True,
    error_model='numpy'
)
def irep_pm(
        p,
        n
):
    """
    Calculate pruning measure for IREP*
    """
    # p + n == 0 is meaningless
    if p + n == 0:
        return -2.0
    else:
        return (p - float(n)) / (p + float(n))


def __last_sequences(
        input_list,
        inverted,
        accept_zero
):
    """
    Last sequences of items.

    Parameters
    ----------
    input_list : list
        Input item list
    inverted : boolean
        If inverted=True, then the resulted sequence starts with full length
    accept_zero : boolean
        If accept_zero=True, then the resulted sequences contains empty

    Returns
    -------
    list
        Last sequences of original list
    """
    if inverted:
        step = -1
        if accept_zero:
            begin = len(input_list) + 1
            end = -1
        else:
            begin = len(input_list) + 1
            end = 0
    else:
        step = 1
        if accept_zero:
            begin = 0
            end = len(input_list) + 1
        else:
            begin = 1
            end = len(input_list) + 1

    return [input_list[: possible_length] for possible_length in range(begin, end, step)]


def __error_rate_rule_list(
        pos,
        neg,
        rule_list
):
    """
        Calculate error rate of rules in pruning data set

        Parameters
        ----------
        pos : const double[::1, :]
            Positive instances of pruning data set
        neg : const double[::1, :]
            Negative instances of pruning data set
        rule_list: list
            Input rule list

        Returns
        -------
        double
            Error rate
        """
    return 1.0 - accuracy_tp_fp(
        len(pos),
        len(neg),
        np.count_nonzero(_rule_list_predict(pos, rule_list)),
        np.count_nonzero(_rule_list_predict(neg, rule_list))
    )


def _pruning_irep(
        pos,
        neg,
        rule
):
    """
    Pruning literals in a rule

    Parameters
    ----------
    pos : const double[::1, :]
        Positive instances of pruning data set
    neg : const double[::1, :]
        Negative instances of pruning data set
    rule: list
        Input rule

    Returns
    -------
    double
        Pruned rule
    """
    return max(
        __last_sequences(rule, True, False),
        key=lambda rule_iter: irep_pm(
            np.count_nonzero(_rule_predict(pos, rule_iter)),
            np.count_nonzero(_rule_predict(neg, rule_iter))
        )
    )


def _minimize_rule_list(
        pos,
        neg,
        rules,
        n
):
    """
    Pruning rules in a rule list

    Parameters
    ----------
    pos : const double[::1, :]
        Positive instances of pruning data set
    neg : const double[::1, :]
        Negative instances of pruning data set
    rules: list
        Input rule list

    Returns
    -------
    double
        Pruned rule list
    """
    return min(
        __last_sequences(rules, True, True),
        key=lambda rule_list_iter: _mdl(pos, neg, rule_list_iter, n)
    )


def _pruning_optimization(
        pos_prune,
        neg_prune,
        rule,
        rules,
        index
):
    """
    Pruning rules in a rule list for the optimization step

    Parameters
    ----------
    pos_prune : const double[::1, :]
        Positive instances of data set
    neg_prune : const double[::1, :]
        Negative instances of data set
    rule: list
        Selected rule
    rules: list
        The rest of rules in that rule list

    Returns
    -------
    double
        Pruned rule
    """
    return min(
        __last_sequences(rule, True, False),
        key=lambda rule_iter: __error_rate_rule_list(
            pos_prune,
            neg_prune,
            rules[:index] + [rule_iter] + rules[index + 1:]
            if len(rule_iter) > 0 else rules[:index] + rules[index + 1:]
        )
    )
