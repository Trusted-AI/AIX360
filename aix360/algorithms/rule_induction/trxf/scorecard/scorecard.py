from numbers import Real
from typing import List, Tuple, Dict, Any
from aix360.algorithms.rule_induction.trxf.core import Feature
from aix360.algorithms.rule_induction.trxf.scorecard import Partition


class Scorecard(object):
    """
    Defines the representation of a scorecard
    """

    def __init__(self, partitions: List[Partition], bias: Real = 0.0):
        """
        @param partitions: A list of instances of class Partition, each specifying the bins for a particular feature.
                           Elements in the list must correspond to distinct features.
        @param bias: The global bias term used during scorecard evaluation
        """
        if len(partitions) == 0:
            raise ValueError('Need len(partitions) > 0 but found len(partitions) == "{}"'.format(len(partitions)))

        # Check that all partitions use different features
        seen = []
        for idx, partition in enumerate(partitions):
            try:
                prev_idx = seen.index(partition.feature)
            except ValueError:
                seen.append(partition.feature)
                continue
            raise ValueError('The instances in partitions must correspond to distinct features but the instance'
                             'at index "{}" and "{}" use the same feature "{}"'
                             .format(prev_idx, idx, partition.feature))

        self._partitions = partitions
        self._bias = bias

    def evaluate(self, assignment: Dict[str, Any]) -> float:
        """
        Computes the output of the scorecard for the given assignment of variables. This output is computed as the
        sum of sub-scores for each partition in the scorecard and the bias term.

        @param assignment: dict mapping variable names to values
        @return: float score computed using the scorecard
        """
        score = self.bias
        for partition in self.partitions:
            score += partition.evaluate(assignment)
        return float(score)

    def __str__(self):
        return '\n'.join([str(partition) for partition in self.partitions] + ['Bias="{}"'.format(self.bias)])

    def __repr__(self):
        return '\n'.join([str(self.__class__)] + [repr(partition) for partition in self.partitions] +
                         ['Bias="{}"'.format(self.bias)])

    @property
    def bias(self) -> Real:
        return self._bias

    @property
    def partitions(self) -> List[Partition]:
        return self._partitions

    @property
    def features(self) -> List[Feature]:
        return [partition.feature for partition in self.partitions]

    @property
    def num_features(self) -> int:
        return len(self.partitions)

    @property
    def bins_per_feature(self) -> List[Tuple[Feature, int]]:
        return [(partition.feature, partition.num_bins) for partition in self.partitions]

    @property
    def total_bins(self) -> int:
        return sum([partition.num_bins for partition in self.partitions])
