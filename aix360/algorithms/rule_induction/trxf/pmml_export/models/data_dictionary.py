try:
    # python >= 3.7
    from dataclasses import dataclass as dataclass
    from dataclasses import field as field
except ImportError:
    from attr import s as dataclass
    from attr import ib as field
import enum
import typing


class DataType(enum.Enum):
    string = 0
    integer = 1
    float = 2
    double = 3
    boolean = 4


class OpType(enum.Enum):
    categorical = 0
    ordinal = 1
    continuous = 2


class Restriction(enum.Enum):
    valid = 0
    invalid = 1
    missing = 2


@dataclass(frozen=True)
class Value:
    value: str = field()
    property: Restriction = field(default=Restriction.valid)


@dataclass(frozen=True)
class DataField:
    name: str = field()
    optype: OpType = field()
    dataType: DataType = field()
    values: typing.Optional[typing.List[Value]] = field(default=None)

    def __post_init__(self):
        if self.values and \
                (self.dataType is not DataType.string or self.optype not in (OpType.ordinal, OpType.categorical)):
            raise ValueError


@dataclass(frozen=True)
class DataDictionary:
    dataFields: typing.List[DataField] = field()
