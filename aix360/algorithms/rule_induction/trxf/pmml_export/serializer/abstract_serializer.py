import abc

from .. import models


class AbstractSerializer(abc.ABC):

    @abc.abstractmethod
    def serialize(self, simple_pmml_ruleset_model: models.SimplePMMLRuleSetModel) -> str:
        raise NotImplementedError
