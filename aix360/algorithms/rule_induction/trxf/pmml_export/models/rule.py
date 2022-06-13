try:
    # python >= 3.7
    from dataclasses import dataclass as dataclass
    from dataclasses import field as field
except ImportError:
    from attr import s as dataclass
    from attr import ib as field
import typing

from .predicate import CompoundPredicate
from .predicate import SimplePredicate

DEFAULT_WEIGHT = 1.0
DEFAULT_CONFIDENCE = 1.0


@dataclass(frozen=True)
class SimpleRule:
    predicate: typing.Union[SimplePredicate, CompoundPredicate] = field()
    score: str = field()

    id: typing.Optional[str] = field(default=None)
    recordCount: typing.Optional[int] = field(default=None)
    nbCorrect: typing.Optional[int] = field(default=None)
    confidence: typing.Optional[float] = field(default=DEFAULT_CONFIDENCE)
    weight: typing.Optional[float] = field(default=DEFAULT_WEIGHT)
