from typing import Any

import pandas as pd

from aix360.algorithms.rule_induction.ripper import RipperExplainer
from aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from aix360.algorithms.rule_induction.trxf.core.ruleset_generator import RuleSetGenerator


class RipperRuleSetGenerator(RuleSetGenerator):
    def generate(self, X: pd.DataFrame, y: pd.Series, to_learn: Any, **kwargs) -> DnfRuleSet:
        """
        Optional kwargs:

        d : int
            The number to bit that a new rule need to gain (default=64)
        k : int
            The number of optimization iteration (default=2)
        pruning_threshold : int
            The minimum number of instances for splitting (default=20)
        random_state:
            The random seed for grow/prune set splitting (default=0)
        """
        ripper = RipperExplainer(**kwargs)
        ripper.fit(X, y, to_learn)
        return ripper.explain()
