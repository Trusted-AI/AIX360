try:
    # python >= 3.7
    from dataclasses import dataclass as dataclass
    from dataclasses import field as field
except ImportError:
    from attr import s as dataclass
    from attr import ib as field
import enum
import typing


class Operator(enum.Enum):
    equal = 0
    notEqual = 1
    lessThan = 2
    lessOrEqual = 3
    greaterThan = 4
    greaterOrEqual = 5
    isMissing = 6


@dataclass(frozen=True)
class SimplePredicate:
    operator: Operator = field()
    field: str = field()
    value: typing.Optional[str] = None


@dataclass(frozen=True)
class TruePredicate:
    pass


# Use functional api to add aliases for `or` and `and`
BooleanOperator = enum.Enum('BooleanOperator',
                            [('or', 0), ('or_', 0), ('and', 1), ('and_', 1), ('xor', 2), ('surrogate', 3)])


@dataclass(frozen=True)
class CompoundPredicate:
    simplePredicates: typing.List[SimplePredicate] = field()
    booleanOperator: BooleanOperator = field()
