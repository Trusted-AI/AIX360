import pandas as pd

from aix360.algorithms.rule_induction.trxf.core.boolean_evaluator import BooleanEvaluator


def set_equality(list1: list, list2: list) -> bool:
    """
    Equivalent to set(list1) == set(list2) but checks the set inequality manually for lists containing elements that are
    not hashable.

    @param list1: The first list
    @param list2: The second list

    @return bool: Whether the two lists are equal when converted to sets
    """

    equality = True
    for item in list1:
        if item not in list2:
            equality = False
            break

    for item in list2:
        if item not in list1:
            equality = False
            break

    return equality


def batch_evaluate(evaluator: BooleanEvaluator, X: pd.DataFrame) -> pd.Series:
    """
    Evaluate the truth value of the evaluator for each row of X, representing the assignments

    @param evaluator: a BooleanEvaluator, e.g., DnfRuleSet, Conjunction, etc.
    @param X: pandas dataframe where column names represent variable names used by the evaluator
    @return: pandas series of boolean values representing
    """
    assignments = X.to_dict('records')
    return pd.Series(list(map(lambda assignment: evaluator.evaluate(assignment), assignments)))
