try:
    # python >= 3.7
    from dataclasses import dataclass as dataclass
    from dataclasses import field as field
except ImportError:
    from attr import s as dataclass
    from attr import ib as field

from .mining_schema import MiningSchema
from .ruleset import RuleSet


@dataclass(frozen=True)
class RuleSetModel:
    miningSchema: MiningSchema = field()
    ruleSet: RuleSet = field()
