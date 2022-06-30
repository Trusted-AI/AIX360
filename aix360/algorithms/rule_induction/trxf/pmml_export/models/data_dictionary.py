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


@dataclass(frozen=True)
class DataField:
    name: str = field()
    optype: OpType = field()
    dataType: DataType = field()


@dataclass(frozen=True)
class DataDictionary:
    dataFields: typing.List[DataField] = field()
