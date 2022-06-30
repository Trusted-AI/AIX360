try:
    # python >= 3.7
    from dataclasses import dataclass as dataclass
    from dataclasses import field as field
except ImportError:
    from attr import s as dataclass
    from attr import ib as field
import enum
import typing


class MiningFieldUsageType(enum.Enum):
    active = 0
    target = 1


@dataclass(frozen=True)
class MiningField:
    name: str = field()
    usageType: MiningFieldUsageType = field(default=MiningFieldUsageType.active)


@dataclass(frozen=True)
class MiningSchema:
    miningFields: typing.List[MiningField] = field()
