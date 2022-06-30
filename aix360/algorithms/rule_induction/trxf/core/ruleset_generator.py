import abc
from typing import Any

import pandas as pd

from aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet


class RuleSetGenerator(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return callable(subclass.generate)

    @abc.abstractmethod
    def generate(self, X: pd.DataFrame, y: pd.Series, to_learn: Any, **kwargs) -> DnfRuleSet:
        """
        Train a rule induction algorithm and generate a trxf DnfRuleSet

        @param X: pandas dataframe representing features
        @param y: pandas series representing labels
        @param to_learn: one of the unique values of y we want to learn, representing the positive class
        @param kwargs: algorithm-specific parameters
        @return: a trxf DnfRuleSet
        """
        raise NotImplementedError
