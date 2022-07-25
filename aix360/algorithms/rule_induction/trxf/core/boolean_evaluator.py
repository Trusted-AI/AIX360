import abc
from typing import Dict, Any


class BooleanEvaluator(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return callable(subclass.evaluate)

    @abc.abstractmethod
    def evaluate(self, assignment: Dict[str, Any]) -> bool:
        """
        Evaluate the truth value w.r.t. the variable assignment
        """
        raise NotImplementedError
