from typing import Dict, List, Tuple

import pandas as pd

from ..base import LocalCounterfactualMethod
from ..utils.action import extract_actions_pandas
from ..utils.centroid import centroid_pandas

def generate_cluster_centroid_explanations(
    cluster_centroids: Dict[int, pd.DataFrame],
    cf_generator: LocalCounterfactualMethod,
    num_local_counterfactuals: int,
    numerical_features_names: List[str],
    categorical_features_names: List[str],
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    cluster_explanations = {
        i: cf_generator.explain_instances(
            cluster_centroids[i], num_local_counterfactuals
        )
        for i, _ in cluster_centroids.items()
    }
    returned_requested = True
    empty_cfs_idxs = []
    for i, cfs in cluster_explanations.items():
        if cfs.empty:
            empty_cfs_idxs.append(i)
        if cfs.shape[0] != num_local_counterfactuals:
            returned_requested = False
    for i in empty_cfs_idxs:
        del cluster_explanations[i]
    
    if not cluster_explanations:
        raise ValueError("No counterfactuals found for any of the centroids.")
    
    if returned_requested:
        cluster_expl_actions = {
            i: extract_actions_pandas(
                X=pd.concat([cluster_centroids[i]] * num_local_counterfactuals).set_index(
                    cluster_explanations[i].index
                ),
                cfs=cluster_explanations[i],
                categorical_features=categorical_features_names,
                numerical_features=numerical_features_names,
                categorical_no_action_token="-",
            )
            for i, _cfs in cluster_explanations.items()
        }
    else:
        cluster_expl_actions = {
            i: extract_actions_pandas(
                X=pd.concat([cluster_centroids[i]] * cluster_explanations[i].shape[0]).set_index(
                    cluster_explanations[i].index
                ),
                cfs=cluster_explanations[i],
                categorical_features=categorical_features_names,
                numerical_features=numerical_features_names,
                categorical_no_action_token="-",
            )
            for i, _cfs in cluster_explanations.items()
        }

    explanations_centroid = {
        i: centroid_pandas(
            X=cluster_explanations[i],
            numerical_columns=numerical_features_names,
            categorical_columns=categorical_features_names,
        )
        for i, _cfs in cluster_explanations.items()
    }

    return cluster_explanations, cluster_expl_actions, explanations_centroid