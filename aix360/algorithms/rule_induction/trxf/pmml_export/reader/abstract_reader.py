import abc
import typing

from .. import models


class AbstractReader(abc.ABC):
    T = typing.TypeVar('T')

    @property
    def data_dictionary(self):
        raise NotImplementedError

    @abc.abstractmethod
    def read(self, model: T) -> models.SimplePMMLRuleSetModel:
        raise NotImplementedError
