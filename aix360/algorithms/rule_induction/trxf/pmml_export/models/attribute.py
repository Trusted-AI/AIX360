import typing

from . import predicate

try:
    # python >= 3.7
    from dataclasses import dataclass as dataclass
    from dataclasses import field as field
except ImportError:
    from attr import s as dataclass
    from attr import ib as field


@dataclass(frozen=True)
class ComplexPartialScore:
    feature_name: str = field()
    multiplier: str = field(default=None)
    constant: str = field(default=None)


@dataclass(frozen=True)
class Attribute:
    score: typing.Union[str, ComplexPartialScore]
    predicate: typing.Union[predicate.SimplePredicate, predicate.CompoundPredicate, predicate.TruePredicate]
