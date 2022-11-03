import abc
import typing

from .. import models


class AbstractSerializer(abc.ABC):

    @abc.abstractmethod
    def serialize(self, model: typing.Union[models.SimplePMMLRuleSetModel, models.Scorecard]) -> str:
        raise NotImplementedError
