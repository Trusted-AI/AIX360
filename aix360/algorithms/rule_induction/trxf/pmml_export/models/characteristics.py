import typing

from . import attribute

try:
    # python >= 3.7
    from dataclasses import dataclass as dataclass
except ImportError:
    from attr import s as dataclass


@dataclass(frozen=True)
class Characteristic:
    name: str
    attributes: typing.List[attribute.Attribute]


@dataclass(frozen=True)
class Characteristics:
    characteristics: typing.List[Characteristic]
