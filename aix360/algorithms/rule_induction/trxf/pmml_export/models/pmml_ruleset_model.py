try:
    # python >= 3.7
    from dataclasses import dataclass as dataclass
    from dataclasses import field as field
except ImportError:
    from attr import s as dataclass
    from attr import ib as field

from .data_dictionary import DataDictionary
from .ruleset_model import RuleSetModel


@dataclass(frozen=True)
class SimplePMMLRuleSetModel:
    dataDictionary: DataDictionary = field()
    ruleSetModel: RuleSetModel = field()
