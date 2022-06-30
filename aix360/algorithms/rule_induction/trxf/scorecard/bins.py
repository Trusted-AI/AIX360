import abc
from numbers import Real
from typing import Dict, Set, Any, Optional
from aix360.algorithms.rule_induction.trxf.core import Feature


class Bin(abc.ABC):
    """
    Defines a set of values of interest for a particular feature along with an associated sub-score.
    """

    def __init__(self, feature: Feature):
        """
        @param feature: An instance of trxf.core.Feature. Specifies the feature with which this bin is associated.
        """
        self._feature = feature

    @abc.abstractmethod
    def evaluate(self, assignment: Dict[str, Any]) -> float:
        """
        Computes the sub-score for the bin based on the feature value.

        @param assignment: dict mapping variable name to value
        @return: float sub-score for the bin
        """
        raise NotImplementedError('Method evaluate in Bin not implemented')

    @abc.abstractmethod
    def contains(self, assignment: Dict[str, Any]) -> bool:
        """
        Checks if the value of the feature belongs to this bin or not.

        @param assignment: dict mapping variable name to value
        @return: bool, true if the feature value belongs to the bin and false otherwise
        """
        raise NotImplementedError('Method contains in Bin not implemented.')

    @abc.abstractmethod
    def overlaps(self, other: 'Bin') -> bool:
        """
        Checks whether this bin overlaps with the given other bin.

        @param other: An instance of the same class as self specifying the other bin
        @return: bool, true if the bins share at least one element and false otherwise
        """
        raise NotImplementedError('Method overlaps in Bin not implemented.')

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError('Method __str__ in Bin not implemented.')

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError('Method __repr__ in Bin not implemented.')

    @property
    def feature(self) -> Feature:
        return self._feature


class LinearIntervalBin(Bin):
    """
    Defines a Bin specified by a half-open interval [left_end, right_end) where the sub-score is calculated using
    a linear function of the feature value (requires the feature to evaluate to a numerical value).
    """

    def __init__(self, feature: Feature, const_part: Real, linear_multiplier: Real, left_end: Optional[Real] = None,
                 right_end: Optional[Real] = None):
        """
        @param feature: An instance of trxf.core.Feature associated with this bin. This must be a numerical feature.
        @param const_part: The constant additive part used in calculating the sub-score.
        @param linear_multiplier: The sub-score will be calculated as linear_multiplier * feature_value + const_part
        @param left_end: The left endpoint of the interval. Set to -infinity if not specified.
        @param right_end: The right endpoint of the interval. Set to +infinity if not specified.
        """
        if left_end is None:
            left_end = float('-inf')
        if right_end is None:
            right_end = float('inf')

        if left_end >= right_end:
            raise ValueError('Need left_end < right_end but left_end = "{}" and right_end = "{}"'
                             .format(left_end, right_end))

        super(LinearIntervalBin, self).__init__(feature)
        self._left_end = left_end
        self._right_end = right_end
        self._linear_multiplier = linear_multiplier
        self._const_part = const_part

    def evaluate(self, assignment: Dict[str, Any]) -> float:
        """
        Computes the sub-score for the bin based on the feature value. The sub-score is zero if the feature value lies
        outside the bin and is given by linear_multiplier * feature_value + const_part otherwise.

        @param assignment: dict mapping variable name to value
        @return: float sub-score for the bin
        """
        if self.contains(assignment):
            return self.linear_multiplier * self._get_feature_value(assignment) + self.const_part
        return 0.0

    def contains(self, assignment: Dict[str, Any]) -> bool:
        """
        Returns true if left_end <= feature_value < right_end for the given variable assignment.

        @param assignment: dict mapping variable name to value
        @return: bool True if the feature value belongs to the bin and False otherwise
        """
        return self.left_end <= self._get_feature_value(assignment) < self.right_end

    def overlaps(self, other: 'LinearIntervalBin') -> bool:
        """
        Checks whether this bin overlaps with the given other bin.

        @param other: An instance of LinearIntervalBin or a subclass thereof
        @return: bool, true if the bins share at least one element and false otherwise
        """
        if not isinstance(other, LinearIntervalBin):
            raise ValueError('other must be an instance of LinearIntervalBin or a subclass thereof but it'
                             'is an instance of "{}"'.format(str(other.__class__)))
        return (self.left_end < other.right_end) and (self.right_end > other.left_end)

    def _get_feature_value(self, assignment: Dict[str, Any]) -> Real:
        """
        Evaluates the value of the feature for the specified variable assignment. Raises ValueError if the feature
        evaluates to a non-numerical value.

        @param assignment: dict mapping variable name to value
        @return: float, value of the feature
        """
        feature_value = self.feature.evaluate(assignment)
        if isinstance(feature_value, Real):
            return feature_value
        raise ValueError('Need the feature to evaluate to a numerical value but the type of returned value is "{}"'
                         .format(type(feature_value)))

    def __str__(self):
        return ' '.join([str(self.left_end), '<=', str(self.feature), '<', str(self.right_end), '->',
                         str(self.linear_multiplier), '*', 'x', '+', str(self.const_part)])

    def __repr__(self):
        return '%s(%r, [%r, %r), %r * x + %r)' % (self.__class__, self.feature, self.left_end, self.right_end,
                                                  self.linear_multiplier, self.const_part)

    @property
    def left_end(self) -> Real:
        return self._left_end

    @property
    def right_end(self) -> Real:
        return self._right_end

    @property
    def linear_multiplier(self) -> Real:
        return self._linear_multiplier

    @property
    def const_part(self) -> Real:
        return self._const_part


class IntervalBin(LinearIntervalBin):
    """
    Defines a bin specified by an interval [left_end, right_end) with a constant sub-score
    """

    def __init__(self, feature: Feature, sub_score: Real, left_end: Optional[Real] = None,
                 right_end: Optional[Real] = None):
        """
        @param feature: An instance of trxf.core.Feature. Specifies the feature to which this bin is associated.
        @param sub_score: The constant sub-score associated with this bin
        @param left_end: The left endpoint on the interval. Set to -infinity if not specified.
        @param right_end: The right endpoint of the interval. Set to +infinity if not specified.
        """
        super(IntervalBin, self).__init__(feature, sub_score, 0.0, left_end, right_end)

    def __str__(self):
        return ' '.join([str(self.left_end), '<=', str(self.feature), '<', str(self.right_end), '->',
                         str(self.const_part)])

    def __repr__(self):
        return '%s(%r, [%r, %r), %r)' % (self.__class__, self.feature, self.left_end, self.right_end, self.const_part)

    @property
    def sub_score(self) -> Real:
        return self.const_part


class SetBin(Bin):
    """
    Defines a Bin specified by a finite set of values and a constant sub-score
    """

    def __init__(self, feature: Feature, sub_score: Real, values: Set[Any]):
        """
        @param feature: An instance of trxf.core.Feature. Specifies the feature to which this bin is associated.
        @param sub_score: The constant sub-score associated with this bin.
        @param values: The set of values that belong to this bin
        """
        if len(values) == 0:
            raise ValueError('Expected len(values) > 0, received len(values) == "{}"'.format(len(values)))

        super(SetBin, self).__init__(feature)
        self._values = values
        self._sub_score = sub_score

    def evaluate(self, assignment: Dict[str, Any]) -> float:
        """
        Computes the sub-score for the bin based on the feature value. The sub-score is zero if the feature value lies
        outside the bin and is given by sub_score otherwise.

        @param assignment: dict mapping variable name to value
        @return: float sub-score for the bin
        """
        if self.contains(assignment):
            return float(self.sub_score)
        return 0.0

    def contains(self, assignment: Dict[str, Any]) -> bool:
        """
        Returns true if the feature value belongs to the set of values in this bin and false otherwise

        @param assignment: dict mapping variable name to value
        @return: bool, true if the feature value belongs to the bin and false otherwise
        """
        return self.feature.evaluate(assignment) in self.values

    def overlaps(self, other: 'SetBin') -> bool:
        """
        Checks whether this bin overlaps with the given other bin.

        @param other: An instance of SetBin or a subclass thereof
        @return: bool, true if the bins share at least one element and false otherwise
        """
        if not isinstance(other, SetBin):
            raise ValueError('other must be an instance of SetBin or a subclass thereof but it'
                             'is an instance of "{}"'.format(str(other.__class__)))
        return len(self.values.intersection(other.values)) > 0

    def __str__(self):
        return ' '.join([str(self.feature), 'in', str(self.values), '->', str(self.sub_score)])

    def __repr__(self):
        return '%s(%r, %r, %r)' % (self.__class__, self.feature, self.values, self.sub_score)

    @property
    def values(self) -> Set[Any]:
        return self._values

    @property
    def sub_score(self) -> Real:
        return self._sub_score
