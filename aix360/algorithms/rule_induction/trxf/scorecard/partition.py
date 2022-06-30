from typing import List, Dict, Any
from aix360.algorithms.rule_induction.trxf.core import Feature
from aix360.algorithms.rule_induction.trxf.scorecard import Bin


class Partition(object):
    """
    A Partition is a set of non-overlapping bins of the same type associated with the same feature
    """

    def __init__(self, bins: List[Bin]):
        """
        @param bins: A list of instances of a subclass of GeneralizedBin. All elements must have the same type.
        """
        if len(bins) == 0:
            raise ValueError('Need len(bins) > 0 but len(bins) = "{}"'.format(len(bins)))

        # Check that all bins are associated with the same feature and have the same type
        self._feature = bins[0].feature
        bin_type = type(bins[0])
        for idx, curr_bin in enumerate(bins):
            if curr_bin.feature != self._feature:
                raise ValueError('All bins must be associated with the same feature but bins[0] was associated with'
                                 '"{}" whereas bins["{}"] was associated with "{}"'.format(self._feature, idx,
                                                                                           curr_bin.feature))
            if type(curr_bin) != bin_type:
                raise ValueError('All bins must have the same type but bins[0] has type "{}" while bins["{}"]'
                                 'has type "{}"'.format(bin_type, idx, type(curr_bin)))

        # Check that bins do not overlap
        for idx1 in range(len(bins)):
            for idx2 in range(idx1 + 1, len(bins)):
                if bins[idx1].overlaps(bins[idx2]):
                    raise ValueError('Bins must be non-overlapping but bin "{}" overlaps with bin "{}"'
                                     .format(bins[idx1], bins[idx2]))

        self._bins = bins

    def evaluate(self, assignment: Dict[str, Any]) -> float:
        """
        Computes the sub-score for the feature associated with this partition. This score is given by the sub-score of
        the first bin in the partition to which the feature value belongs. Raises ValueError if the feature value does
        not belong to any of the bins.

        @param assignment: dict mapping variable name to value
        @return: float sub-score associated with this partition.
        """
        for curr_bin in self.bins:
            if curr_bin.contains(assignment):
                return curr_bin.evaluate(assignment)
        raise ValueError('Value "{}" of the feature does not belong to any of the bins'
                         .format(self._feature.evaluate(assignment)))

    def __str__(self):
        return '\n'.join([str(self.feature)] + ['\t' + str(curr_bin) for curr_bin in self.bins])

    def __repr__(self):
        return '\n'.join([str(self.__class__)] + ['\t' + repr(self.feature)] +
                         ['\t' + repr(curr_bin) for curr_bin in self.bins])

    @property
    def num_bins(self) -> int:
        return len(self.bins)

    @property
    def bins(self) -> List[Bin]:
        return self._bins

    @property
    def feature(self) -> Feature:
        return self._feature
