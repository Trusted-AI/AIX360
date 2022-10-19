import typing

from . import characteristics
from . import data_dictionary
from . import mining_schema

try:
    # python >= 3.7
    from dataclasses import dataclass as dataclass
    from dataclasses import field as field
except ImportError:
    from attr import s as dataclass
    from attr import ib as field


@dataclass(frozen=True)
class OutputField:
    name: str
    feature: str
    dataType: data_dictionary.DataType
    optype: data_dictionary.OpType


@dataclass(frozen=True)
class Output:
    outputFields: typing.List[OutputField]


@dataclass(frozen=True)
class Scorecard:
    dataDictionary: data_dictionary.DataDictionary
    miningSchema: mining_schema.MiningSchema
    output: Output
    characteristics: characteristics.Characteristics
    initialScore: str = field(default="0")
