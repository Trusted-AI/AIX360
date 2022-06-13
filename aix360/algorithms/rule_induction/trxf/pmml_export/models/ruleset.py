try:
    # python >= 3.7
    from dataclasses import dataclass as dataclass
    from dataclasses import field as field
except ImportError:
    from attr import s as dataclass
    from attr import ib as field
import enum
import typing

from .rule import SimpleRule


class RuleSelectionMethod(enum.Enum):
    firstHit = 0
    weightedMax = 1
    weightedSum = 2


@dataclass(frozen=True)
class RuleSet:
    ruleSelectionMethod: typing.List[RuleSelectionMethod] = field()
    rules: typing.List[SimpleRule] = field()

    recordCount: typing.Optional[int] = field(default=None)
    nbCorrect: typing.Optional[int] = field(default=None)
    defaultScore: typing.Optional[str] = field(default=None)
    defaultConfidence: typing.Optional[float] = field(default=None)
