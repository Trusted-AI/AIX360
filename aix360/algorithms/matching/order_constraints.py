
import numpy as np

from aix360.algorithms.lwbe import LocalWBExplainer

from typing import Tuple, Optional, List
from typing import NamedTuple

Index = Tuple[int,int]


class AlternateMatching(NamedTuple):

    """
    OTMatchingExplainer returns an ordered list of objects, 
    each repreenting an explaination.

    Attributes:

        matching (numpy 2d array): alternative matching
        salient (list of int tuples): salient matches (i,j) that constrast with the explained matching
    """

    matching: np.ndarray
    salient: List[Tuple]

class OTMatchingExplainer(LocalWBExplainer):

    """
    OTMatchingExplainer provides explainations for a matching
    that satisfies the transport polytope constraints.
    Given a matching, it produces a set of alternative matchings, 
    where each alternate contrasts with the provided instance 
    by a sparse set of salient matchings. [#]_.

    This is akin to a search engine providing alternative suggestions
    relevant to a search string. OTMatchingExplainer aims to provide
    the same for matchings.

    References:
        .. [#] `Fabian Lim, Laura Wynter, Shiau Hong Lim, 
           "Order Constraints in Optimal Transport", 
           2022
           <https://arxiv.org/abs/2110.07275>`_
    """

    def __init__(
        self, 
        deactivate_bounds: bool = False,
        error_limit : float = 1e-3,
    ):
        """
        Initialize the OTMatchingExplainer
        """

        import sys
        if sys.version_info.major == 2:
            super(OTMatchingExplainer, self).__init__()
        else:
            super().__init__()

        self._deactivate_bounds = deactivate_bounds
        self._error_limit = error_limit

    def set_params(self, *args, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def explain_instance(
        self, 
        matching: np.ndarray, 
        costs: np.ndarray,
        constraints: Tuple[
            np.ndarray, 
            np.ndarray, 
        ],
        num_alternate_matchings: int = 1,
        search_thresholds: Tuple[float, float] = (.5, .5),
        search_node_limit: int = 20,
        search_depth_limit: int = 1,
        search_match_pos_filter: Optional[List[Index]]=None,
    ):
        """
        Explain the matching 

        Args:
            matching (numpy 2d array): the matching to be explained.
            costs: (numpy 2d array):  the (non-negative) matching costs used to obtain above matching.
            constraints (numpy array, numpy array): the transport polytope row/column constraints.
            num_alternate_matchings (int): the number of alternate matchings to return back.
            search_node_limit (int): stop the search when this many nodes have been encountered.
            search_depth_limit (int): do not progress beyond this tree level in the search
            search_match_pos_filter ((int,int) array or None): if specified, this is a whitelist of positions (i,j) of candidate match positions
            search_thresholds (float, float): thresholds used to pick the candidate match positions to search over.

        Returns: 
            list of AlternateMatching explanations.
        """

        # the row and column constraints
        a, b = constraints

        # check the filter
        if search_match_pos_filter is not None:
            for x in search_match_pos_filter:
                if (
                    (len(x) != 2)
                    or 
                    (type(x) != tuple)
                ):
                    raise ValueError(f"search_match_pos_filter must only contain 2-tuples")

        # TODO: remove the warnings here when the "import numpy.matlib"
        # issue has been resoluved.
        import warnings
        with warnings.catch_warnings(record=True):
            from otoc import search_otoc_candidates2

        self._model = search_otoc_candidates2(
            a, b, costs,
            strategy = (
                'least-saturated-coef',
                {
                    'base_solution': matching, 
                    'saturationThreshold': search_thresholds,
                    'a': a,
                    'b': b,
                    'index_filter': search_match_pos_filter,
                }
            ),
            numCandidates=num_alternate_matchings,
            limitCandidates=search_node_limit,
            limitCandatesMode='candidates-obtained',
            limitDepth=search_depth_limit,
            deactivate_bounds=self._deactivate_bounds,
            acceptableError=self._error_limit,
        )

        # perform the search to get various match candidates
        for algo in self._model:
            for _ in algo:
                pass

        # search history
        history = self._model._history

        # return the top candidate matches
        results = []
        for i in range(1, num_alternate_matchings+1):
            x = self._model.best_solution(n=i)

            # if None is returned, then the search has
            # terminated early and there are a deficit of 
            # candidates.
            # So then just terminate here
            
            if x is None:
                break

            results.append(
                AlternateMatching(
                    matching=x,
                    salient=history[
                        self._model.best_history_index(n=i)
                    ].index, # type: ignore
                )
            )
        return results
