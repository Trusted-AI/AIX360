import numba as nb
import numpy as np

from aix360.algorithms.rule_induction.ripper.binding import _rule_list_predict


@nb.njit(
    'double(int64, int64)',
    nogil=True,
    cache=True,
    error_model='numpy'
)
def __theorem_length(
        k,
        n
):
    """
    Calculate the MDL for a rule

    Parameters
    ----------
    k: uint64_t
        Literal count for theorem
    n : uint64_t
        Total literal count

    Returns
    -------
    double
        The MDL for a rule
    """
    if k == 0:
        return 0.0
    p = k / float(n)
    return np.log2(k) + k * np.log2(1 / p) + (n - k) * np.log2(1 / (1 - p))


@nb.njit(
    'double(int64, int64)',
    nogil=True,
    cache=True,
    error_model='numpy'
)
def __msg_length_uniform(
        e,
        d
):
    """
    Calculate the MDL a data set

    Parameters
    ----------
    e: uint64_t
        Error count
    d : uint64_t
        Total number of instances

    Returns
    -------
    double
        The MDL for data set
    """
    if e == 0 or e == d:
        return np.log2(d + 1)
    return np.log2(d + 1) + e * (-np.log2(e / float(d))) + (d - e) * (-np.log2(1 - (e / float(d))))


def _mdl(
        pos,
        neg,
        rule_list,
        n
):
    """
    Calculate the MDL for a rule list and the training data from which the rule list is learned

    Parameters
    ----------
    pos : const double[::1, :]
        Positive instances of training data
    neg : const double[::1, :]
        Negative instances of training data
    rule_list : list
        Learned rule list
    n : uint64_t
        Total literal count

    Returns
    -------
    double
        The MDL for the rule list and the training data from which the rule list is learned
    """
    e = np.count_nonzero(_rule_list_predict(neg, rule_list)) + \
        len(pos) - np.count_nonzero(_rule_list_predict(pos, rule_list))

    return 0.5 * np.sum([__theorem_length(len(rule), n) for rule in rule_list]) + \
        __msg_length_uniform(e, len(pos) + len(neg))
