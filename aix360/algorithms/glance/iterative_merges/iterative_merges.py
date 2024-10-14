from typing import Union, Any, List, Optional, Dict, Tuple, Callable, Literal
import math
import numbers
import itertools
from tqdm import tqdm
import warnings
from colorama import Fore, Style

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import DisjointSet
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from IPython.display import display
from ..base import GlobalCounterfactualMethod
from ..base import LocalCounterfactualMethod
from ..base import ClusteringMethod
from ..utils.centroid import centroid_pandas
from ..utils.action import (
    apply_action_pandas,
    actions_mean_pandas,
)
from ..counterfactual_costs import build_dist_func_dataframe
from ..utils.metadata_requests import _decide_cluster_method, _decide_local_cf_method
from .phase2 import generate_cluster_centroid_explanations


class IterativeMerges(GlobalCounterfactualMethod):

    def __init__(
        self,
        model: Any,
        initial_clusters: int = 100,
        final_clusters: int = 10,
        num_local_counterfactuals: int = 5,
        heuristic_weights: Tuple[float, float] = (0.5, 0.5),
        alternative_merges: bool = True,
        random_seed: int = 13,
        verbose=True,
    ) -> None:
        super().__init__()
        self.model = model
        self.initial_clusters = initial_clusters
        self.final_clusters = final_clusters
        self.num_local_counterfactuals = num_local_counterfactuals
        self.heuristic_weights = heuristic_weights
        self.alternative_merges = alternative_merges
        self.random_seed = random_seed
        self.verbose = verbose
        self.final_clustering = None
        self.clusters_results = None

    def _set_features_names(
        self,
        X: pd.DataFrame,
        numerical_names: Optional[List[str]],
        categorical_names: Optional[List[str]]
    ) -> Tuple[List[str], List[str]]:
        if numerical_names is None and categorical_names is None:
            numerical_names = X.select_dtypes(
                include=["number"]
            ).columns.tolist()
            categorical_names = X.columns.difference(
                numerical_names
            ).tolist()
        elif numerical_names is None and categorical_names is not None:
            numerical_names = X.columns.difference(categorical_names).tolist()
        elif numerical_names is not None and categorical_names is None:
            categorical_names = X.columns.difference(numerical_names).tolist()
        
        assert numerical_names is not None and categorical_names is not None
        return numerical_names, categorical_names
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_dataset: pd.DataFrame,
        feat_to_vary: Optional[Union[List[str], str]] = "all",
        numeric_features_names: Optional[List[str]] = None,
        categorical_features_names: Optional[List[str]] = None,
        clustering_method: Union[ClusteringMethod, Literal["KMeans"]] = "KMeans",
        cf_generator: Union[
            LocalCounterfactualMethod,
            Literal["Dice", "NearestNeighbors", "RandomSampling"]
        ] = "Dice",
        cluster_action_choice_algo: Literal["max-eff", "mean-act", "low-cost", "min-cost-eff-thres", "eff-thres-hybrid"] = "max-eff",
        nns__n_scalars: Optional[int] = None,
        rs__n_most_important: Optional[int] = None,
        rs__n_categorical_most_frequent: Optional[int] = None,
        lowcost__action_threshold: Optional[int] = None,
        lowcost__num_low_cost: Optional[int] = None,
        min_cost_eff_thres__effectiveness_threshold: Optional[float] = None,
        min_cost_eff_thres_combinations__num_min_cost: Optional[int] = None,
        eff_thres_hybrid__max_n_actions_full_combinations: Optional[int] = None,
    ) -> "IterativeMerges":
        self.numerical_features_names, self.categorical_features_names = self._set_features_names(
            X=X,
            numerical_names=numeric_features_names,
            categorical_names=categorical_features_names,
        )

        self.X = X
        self.y = y
        self.train_dataset = train_dataset
        self.clustering_method_ = clustering_method
        self.action_threshold = lowcost__action_threshold if lowcost__action_threshold is not None else 1.5
        self.num_low_cost = lowcost__num_low_cost if lowcost__num_low_cost is not None else 20
        self.effectiveness_threshold = min_cost_eff_thres__effectiveness_threshold if min_cost_eff_thres__effectiveness_threshold is not None else 0.1
        self.min_cost_eff_thres_combinations__num_min_cost = min_cost_eff_thres_combinations__num_min_cost
        self.cluster_action_choice_algo: Literal["max-eff", "mean-act", "low-cost", "min-cost-eff-thres", "min-cost-eff-thres-combinations", "hybrid"] = cluster_action_choice_algo
        self.eff_thres_hybrid__max_n_actions_full_combinations = eff_thres_hybrid__max_n_actions_full_combinations if eff_thres_hybrid__max_n_actions_full_combinations is None else 50
        
        if nns__n_scalars is not None:
            self.n_scalars = nns__n_scalars
        else:
            self.n_scalars = 1000
        if rs__n_most_important is not None:
            self.n_most_important = rs__n_most_important
        else:
            self.n_most_important = len(X.columns)
        if rs__n_categorical_most_frequent is not None:
            self.n_categorical_most_frequent = rs__n_categorical_most_frequent
        else:
            self.n_categorical_most_frequent = 20
        
        self.cf_generator = _decide_local_cf_method(
            method=cf_generator,
            model=self.model,
            train_dataset=self.train_dataset,
            numeric_features_names=self.numerical_features_names,
            categorical_features_names=self.categorical_features_names,
            feat_to_vary=feat_to_vary,
            random_seed=self.random_seed,
            n_scalars=self.n_scalars,
            n_most_important=self.n_most_important,
            n_categorical_most_frequent=self.n_categorical_most_frequent,
        )

        self.dist_func_dataframe = build_dist_func_dataframe(
                X=X,
                numerical_columns=self.numerical_features_names,
                categorical_columns=self.categorical_features_names,
            )
        return self

    def explain_group(
        self, instances: pd.DataFrame
    ) -> Tuple[int, float]:
        if self.initial_clusters > instances.shape[0]:
            warnings.warn(
                "Requested number of initial clusters is larger than the number of instances to explain. Setting to number of instances."
            )
            self.initial_clusters = instances.shape[0]

        self.clustering_method = _decide_cluster_method(
            self.clustering_method_, self.initial_clusters, self.random_seed
        )

        clusters = _generate_clusters(
            instances=instances,
            num_clusters=self.initial_clusters,
            categorical_features_names=self.categorical_features_names,
            clustering_method=self.clustering_method,
        )

        cluster_centroids = {
            i: centroid_pandas(
                X=instances,
                numerical_columns=self.numerical_features_names,
                categorical_columns=self.categorical_features_names,
            )
            for i, instances in clusters.items()
        }

        cluster_explanations, cluster_expl_actions, explanations_centroid = (
            generate_cluster_centroid_explanations(
                cluster_centroids=cluster_centroids,
                cf_generator=self.cf_generator,
                num_local_counterfactuals=self.num_local_counterfactuals,
                numerical_features_names=self.numerical_features_names,
                categorical_features_names=self.categorical_features_names,
            )
        )
        # delete clusters with no explanations
        clusters = {i: cluster for i, cluster in clusters.items() if i in cluster_explanations.keys()}
        cluster_centroids = {i: cluster for i, cluster in cluster_centroids.items() if i in cluster_explanations.keys()}

        while len(clusters) > self.final_clusters:
            cluster1, cluster2 = _find_candidate_clusters(
                clusters=clusters,
                cluster_centroids=cluster_centroids,
                explanations_centroid=explanations_centroid,
                heuristic_weights=self.heuristic_weights,
                dist_func_dataframe=self.dist_func_dataframe,
            )

            _merge_clusters(
                cluster1=cluster1,
                cluster2=cluster2,
                clusters=clusters,
                cluster_explanations=cluster_explanations,
                cluster_centroids=cluster_centroids,
                cluster_expl_actions=cluster_expl_actions,
                explanations_centroid=explanations_centroid,
                numerical_features_names=self.numerical_features_names,
                categorical_features_names=self.categorical_features_names,
            )

        clusters_res, total_eff, total_cost = cluster_results(
            model=self.model,
            instances=instances,
            clusters=clusters,
            cluster_expl_actions=cluster_expl_actions,
            dist_func_dataframe=self.dist_func_dataframe,
            numerical_features_names=self.numerical_features_names,
            categorical_features_names=self.categorical_features_names,
            cluster_action_choice_algo=self.cluster_action_choice_algo,
            action_threshold=self.action_threshold,
            num_low_cost=self.num_low_cost,
            effectiveness_threshold=self.effectiveness_threshold,
            num_min_cost=self.min_cost_eff_thres_combinations__num_min_cost,
            max_n_actions_full_combinations=self.eff_thres_hybrid__max_n_actions_full_combinations,
        )

        for i, stats in clusters_res.items():
            stats["size"] = clusters[i].shape[0]

        if self.verbose == True:
            format_glance_output(
                cluster_stats=clusters_res,
                categorical_columns = self.categorical_features_names)
#             print_results(
#                 clusters_stats=clusters_res,
#                 total_effectiveness=total_eff,
#                 total_cost=total_cost,
#             )
            
        eff, cost = cumulative(
            self.model,
            instances,
            [stats["action"] for i, stats in clusters_res.items()],
            self.dist_func_dataframe,
            self.numerical_features_names,
            self.categorical_features_names,
            "-",
        )
        if self.verbose == True:
            print(f"{Style.BRIGHT}TOTAL EFFECTIVENESS:{Style.RESET_ALL} {Fore.GREEN}{eff / instances.shape[0]:.2%}{Fore.RESET}")
            print(f"{Style.BRIGHT}TOTAL COST:{Style.RESET_ALL} {Fore.MAGENTA}{(cost / eff):.2f}{Fore.RESET}")

        self.final_clustering = clusters
        self.cluster_results = clusters_res

        return eff, cost

    def global_actions(self):
        return [stats["action"] for i, stats in self.cluster_results.items()]


def cumulative(
    model,
    instances,
    actions,
    dist_func_dataframe,
    numeric_features_names,
    categorical_features_names,
    categorical_no_action_token,
):
    costs = []
    all_predictions = []

    for action in actions:
        applied_df = apply_action_pandas(
            instances,
            action,
            numeric_features_names,
            categorical_features_names,
            categorical_no_action_token,
        )

        predictions = model.predict(applied_df)
        all_predictions.append(predictions)
        cur_costs = dist_func_dataframe(instances.reset_index(drop=True), applied_df.reset_index(drop=True))
#         import pdb
#         pdb.set_trace()
        cur_costs[predictions == 0] = np.inf
        costs.append(cur_costs)

    if costs == []:
        return 0, 0.
    final_costs = np.column_stack(costs).min(axis=1)
    effectiveness = (final_costs != np.inf).sum()
    cost = final_costs[final_costs != np.inf].sum()

    return effectiveness, cost


def action_fake_cost(
    action: pd.Series,
    numerical_features_names: List[str],
    categorical_features_names: List[str],
):
    return (
        action[numerical_features_names].sum()
        + (action[categorical_features_names] != "-").sum()
    )


def _select_action_low_cost(
    model: Any,
    instances: pd.DataFrame,
    cluster_instances: pd.DataFrame,
    candidate_actions: pd.DataFrame,
    dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    numerical_features_names: List[str],
    categorical_features_names: List[str],
    action_threshold: int,
    num_low_cost: int,
    inv_total_clusters: int,
):
    actions_list = [action for _, action in candidate_actions.iterrows()]
    actions_list.sort(
        key=lambda action: action_fake_cost(
            action, numerical_features_names, categorical_features_names
        )
    )
    cf_list = []
    for action in tqdm(
        actions_list[: min(num_low_cost, len(actions_list))], total=len(actions_list)
    ):
        cfs = apply_action_pandas(
            X=instances,
            action=action,
            numerical_columns=numerical_features_names,
            categorical_columns=categorical_features_names,
            categorical_no_action_token="-",
        )
        predictions: np.ndarray = model.predict(cfs)
        n_flipped = predictions.sum()

        if n_flipped > (action_threshold * inv_total_clusters) * len(instances):
            cfs = apply_action_pandas(
                X=cluster_instances,
                action=action,
                numerical_columns=numerical_features_names,
                categorical_columns=categorical_features_names,
                categorical_no_action_token="-",
            )
            predictions: np.ndarray = model.predict(cfs)
            n_flipped = predictions.sum()
            factuals_flipped = cluster_instances[predictions == 1]
            cfs_flipped = cfs[predictions == 1]
            recourse_cost_sum = dist_func_dataframe(factuals_flipped, cfs_flipped).sum()
            cf_list.append((n_flipped, recourse_cost_sum, action))

    if len(cf_list) == 0:
        raise ValueError(
            "Change action_threshold. No action found in cluster with effectiveness in all instances above the threshold"
        )
    else:
        n_flipped, min_recourse_cost_sum, best_action = min(
            cf_list, key=lambda x: (x[1], -x[0])
        )

        return n_flipped, min_recourse_cost_sum, best_action


# def _select_action_min_cost_eff_thres(
#     model: Any,
#     instances: pd.DataFrame,
#     cluster_instances: pd.DataFrame,
#     candidate_actions: pd.DataFrame,
#     dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
#     numerical_features_names: List[str],
#     categorical_features_names: List[str],
#     effectiveness_threshold: int,
# ):
#     actions_list = [action for _, action in candidate_actions.iterrows()]
#     actions_list.sort(
#         key=lambda action: action_fake_cost(
#             action, numerical_features_names, categorical_features_names
#         )
#     )
#     cf_list = []
#     for action in tqdm(
#         actions_list, total=len(actions_list)
#     ):
#         cfs = apply_action_pandas(
#             X=instances,
#             action=action,
#             numerical_columns=numerical_features_names,
#             categorical_columns=categorical_features_names,
#             categorical_no_action_token="-",
#         )
#         predictions: np.ndarray = model.predict(cfs)
#         n_flipped = predictions.sum()

#         if n_flipped / len(instances) >= effectiveness_threshold:
#             cfs = apply_action_pandas(
#                 X=cluster_instances,
#                 action=action,
#                 numerical_columns=numerical_features_names,
#                 categorical_columns=categorical_features_names,
#                 categorical_no_action_token="-",
#             )
#             predictions: np.ndarray = model.predict(cfs)
#             n_flipped = predictions.sum()
#             factuals_flipped = cluster_instances[predictions == 1]
#             cfs_flipped = cfs[predictions == 1]
#             recourse_cost_sum = dist_func_dataframe(factuals_flipped, cfs_flipped).sum()
#             cf_list.append((n_flipped, recourse_cost_sum, action))

#     if len(cf_list) == 0:
#         raise ValueError(
#             "Change action_threshold. No action found in cluster with effectiveness in all instances above the threshold"
#         )
#     else:
#         n_flipped, min_recourse_cost_sum, best_action = min(
#             cf_list, key=lambda x: (x[1], -x[0])
#         )

#         return n_flipped, min_recourse_cost_sum, best_action


def actions_cumulative_eff_cost(
    model: Any,
    X: pd.DataFrame,
    actions_with_costs: List[Tuple[pd.Series, float]],
    dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    numerical_columns: List[str],
    categorical_columns: List[str],
    categorical_no_action_token: Any,
) -> Tuple[float, float]:
    X = X.copy()
    actions_with_costs = sorted(actions_with_costs, key=lambda t: t[1])
    n_flipped_total = 0
    recourse_cost_sum = 0
    for action, _old_cost in actions_with_costs:
        cfs = apply_action_pandas(
            X=X,
            action=action,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            categorical_no_action_token=categorical_no_action_token,
        )
        predictions: np.ndarray = model.predict(cfs)
        n_flipped_total += predictions.sum()
        factuals_flipped = X[predictions == 1]
        cfs_flipped = cfs[predictions == 1]
        recourse_cost_sum += dist_func_dataframe(factuals_flipped, cfs_flipped).sum()
        X = X[predictions == 0]

    return n_flipped_total, recourse_cost_sum

# def _select_action_min_cost_eff_thres_combinations(
#     model: Any,
#     instances: pd.DataFrame,
#     clusters: Dict[int, pd.DataFrame],
#     candidate_actions: Dict[int, pd.DataFrame],
#     dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
#     numerical_features_names: List[str],
#     categorical_features_names: List[str],
#     effectiveness_threshold: float,
#     num_min_cost: Optional[int] = None,
# ):
#     actions_list = [action for actions_cluster in candidate_actions.values() for _, action in actions_cluster.iterrows()]
#     actions_list_with_cost = []
#     for action in tqdm(actions_list):
#         cfs = apply_action_pandas(
#             X=instances,
#             action=action,
#             numerical_columns=numerical_features_names,
#             categorical_columns=categorical_features_names,
#             categorical_no_action_token="-",
#         )
#         predictions: np.ndarray = model.predict(cfs)
#         n_flipped = predictions.sum()
#         factuals_flipped = instances[predictions == 1]
#         cfs_flipped = cfs[predictions == 1]
#         mean_recourse_cost = dist_func_dataframe(factuals_flipped, cfs_flipped).mean()
#         actions_list_with_cost.append((action, mean_recourse_cost))
    
#     actions_list_with_cost.sort(key=lambda t: t[1])
#     if num_min_cost is not None:
#         actions_list_with_cost = actions_list_with_cost[:num_min_cost]
    
#     num_actions = len(clusters)
    
#     best_action_set = None
#     for candidate_action_set in itertools.combinations(actions_list_with_cost, num_actions):
#         n_flipped, cost_sum = actions_cumulative_eff_cost(
#             model=model,
#             X=instances,
#             actions_with_costs=list(candidate_action_set),
#             dist_func_dataframe=dist_func_dataframe,
#             numerical_columns=numerical_features_names,
#             categorical_columns=categorical_features_names,
#             categorical_no_action_token="-",
#         )
    
#         if n_flipped >= effectiveness_threshold * instances.shape[0]:
#             if best_action_set is None or cost_sum < best_cost_sum:
#                 best_action_set = candidate_action_set
#                 best_n_flipped = n_flipped
#                 best_cost_sum = cost_sum

#     if best_action_set is None:
#         raise ValueError(
#             "Change effectiveness_threshold. No action set found with cumulative effectiveness above the threshold"
#         )
#     else:
#         return best_n_flipped, best_cost_sum, [p[0] for p in best_action_set]


# def _select_actions_eff_thres_hybrid(
#     model: Any,
#     instances: pd.DataFrame,
#     clusters: Dict[int, pd.DataFrame],
#     candidate_actions: Dict[int, pd.DataFrame],
#     dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
#     numerical_features_names: List[str],
#     categorical_features_names: List[str],
#     effectiveness_threshold: float,
#     max_n_actions_full_combinations: int = 10,
# ):
#     actions_list = [action for actions_cluster in candidate_actions.values() for _, action in actions_cluster.iterrows()]
#     action_individual_costs = np.empty((instances.shape[0], len(actions_list)))
#     for i, action in enumerate(tqdm(actions_list)):
#         cfs = apply_action_pandas(
#             X=instances,
#             action=action,
#             numerical_columns=numerical_features_names,
#             categorical_columns=categorical_features_names,
#             categorical_no_action_token="-",
#         )
#         predictions: np.ndarray = model.predict(cfs)
#         action_individual_costs[predictions == 0, i] = np.inf

#         factuals_flipped = instances[predictions == 1]
#         cfs_flipped = cfs[predictions == 1]
#         individual_recourse_costs = dist_func_dataframe(factuals_flipped, cfs_flipped)
#         action_individual_costs[predictions == 1, i] = individual_recourse_costs
    
#     dominated = np.zeros(action_individual_costs.shape[1])
#     for i in tqdm(range(action_individual_costs.shape[1])):
#         for j in range(i, action_individual_costs.shape[1]):
#             if (action_individual_costs[:, i] <= action_individual_costs[:, j]).all() and (action_individual_costs[:, i] < action_individual_costs[:, j]).any():
#                 dominated[j] = 1
#             if (action_individual_costs[:, i] >= action_individual_costs[:, j]).all() and (action_individual_costs[:, i] > action_individual_costs[:, j]).any():
#                 dominated[i] = 1

#     action_individual_costs = action_individual_costs[:, ~dominated.astype(bool)]
#     non_dominated_idxs = np.where(dominated == 0)[0]
#     actions_list = [action for i, action in enumerate(actions_list) if i in non_dominated_idxs]
    
#     uf = DisjointSet(range(action_individual_costs.shape[1]))
#     for i in tqdm(range(action_individual_costs.shape[1])):
#         for j in range(i, action_individual_costs.shape[1]):
#             if not uf.connected(i, j):
#                 if (action_individual_costs[:, i] == action_individual_costs[:, j]).all():
#                     uf.merge(i, j)
    
#     sufficient_actions_idxs = [eq_class.pop() for eq_class in uf.subsets()]
#     actions_list = [actions_list[i] for i in sufficient_actions_idxs]
#     action_individual_costs = action_individual_costs[:, sufficient_actions_idxs]
    
#     naned_action_individual_costs = np.where(np.isinf(action_individual_costs), np.nan, action_individual_costs)
#     action_cost_means = np.nanmean(naned_action_individual_costs, axis=0)
#     action_n_flipped = (action_individual_costs != np.inf).astype(int).sum(axis=0)
    
#     costs_sorted_idxs = np.argsort(action_cost_means)
#     effs_sorted_idxs = np.argsort(action_n_flipped)
#     effs_over_costs_sorted_idx = np.argsort(action_n_flipped / action_cost_means)
#     n_slice = max_n_actions_full_combinations // 6
    
#     # Get indices of the n_slice smallest costs
#     smallest_cost_indices = costs_sorted_idxs[:n_slice]
#     # Get indices of the n_slice largest numbers of flipped individuals
#     largest_eff_indices = effs_sorted_idxs[-n_slice:]
#     # Get indices of the n_slice middle cost values
#     mid_start = (len(costs_sorted_idxs) - n_slice) // 2  # Start position of the middle n_slice values
#     middle_cost_indices = costs_sorted_idxs[mid_start:mid_start + n_slice]
#     # Get indices of the n_slice middle n_flipped values
#     mid_start = (len(effs_sorted_idxs) - n_slice) // 2  # Start position of the middle n_slice values
#     middle_eff_indices = effs_sorted_idxs[mid_start:mid_start + n_slice]
#     # Get indices of the n_slice largest n_flipped / cost_mean
#     largest_ratio_indices = effs_over_costs_sorted_idx[-n_slice:]
#     # Finally, get n_slice random indices
#     random_indices = np.random.choice(list(range(len(action_cost_means))), n_slice)
    
#     candidate_idxs = set(smallest_cost_indices) | set(largest_eff_indices) | set(middle_cost_indices) | set(middle_eff_indices) | set(largest_ratio_indices) | set(random_indices)
#     candidate_idxs = np.array(list(candidate_idxs))
    
#     num_actions = len(clusters)
#     best_action_set = None
#     for candidate_action_set in tqdm(itertools.combinations(candidate_idxs, num_actions), total=math.comb(len(candidate_idxs), num_actions)):
#         cand_matrix = action_individual_costs[:, candidate_action_set]
#         min_individual_costs = cand_matrix.min(axis=1)
#         n_individuals = min_individual_costs.shape[0]

#         n_flipped = (np.where(min_individual_costs != np.inf))[0].shape[0]
#         effectiveness = n_flipped / n_individuals
#         cost_sum = min_individual_costs[min_individual_costs != np.inf].sum()

#         if effectiveness >= effectiveness_threshold:
#             if best_action_set is None or cost_sum < best_cost_sum:
#                 best_action_set = candidate_action_set
#                 best_cost_sum = cost_sum
#                 best_n_flipped = n_flipped
    
#     if best_action_set is None:
#         raise ValueError(
#             "Change effectiveness_threshold. No action set found with cumulative effectiveness above the threshold"
#         )
#     else:
#         return best_n_flipped, best_cost_sum, [actions_list[i] for i in best_action_set]


def _select_action_max_eff(
    model: Any,
    instances: pd.DataFrame,
    candidate_actions: pd.DataFrame,
    dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    numerical_features_names: List[str],
    categorical_features_names: List[str],
    num_actions: int = 1,
) -> Tuple[int, int, pd.Series]:
    max_n_flipped = 0
    cf_list = []

    for _, action in tqdm(
        candidate_actions.iterrows(), total=candidate_actions.shape[0]
    ):
        cfs = apply_action_pandas(
            X=instances,
            action=action,
            numerical_columns=numerical_features_names,
            categorical_columns=categorical_features_names,
            categorical_no_action_token="-",
        )
        predictions: np.ndarray = model.predict(cfs)
        n_flipped = predictions.sum()

        if n_flipped < max_n_flipped and num_actions == 1:
            continue
        max_n_flipped = n_flipped

        factuals_flipped = instances[predictions == 1]
        cfs_flipped = cfs[predictions == 1]
        recourse_cost_sum = dist_func_dataframe(factuals_flipped, cfs_flipped).sum()
        cf_list.append((n_flipped, recourse_cost_sum, action))

    if num_actions == 1:
        max_n_flipped, recourse_cost_sum, best_action = max(
            cf_list, key=lambda x: (x[0], -x[1])
        )

        return max_n_flipped, recourse_cost_sum, best_action
    else:
        cf_list.sort(key=lambda x: (-x[0], x[1]))
        return cf_list[:num_actions]


def _select_action_mean(
    model: Any,
    instances: pd.DataFrame,
    candidate_actions: pd.DataFrame,
    dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    numerical_features_names: List[str],
    categorical_features_names: List[str],
) -> Tuple[int, int, pd.Series]:
    mean_action = actions_mean_pandas(
        actions=candidate_actions,
        numerical_features=numerical_features_names,
        categorical_features=categorical_features_names,
        categorical_no_action_token="-",
    )
    cfs = apply_action_pandas(
        X=instances,
        action=mean_action,
        numerical_columns=numerical_features_names,
        categorical_columns=categorical_features_names,
        categorical_no_action_token="-",
    )
    predictions: np.ndarray = model.predict(cfs)
    n_flipped = predictions.sum()
    factuals_flipped = instances[predictions == 1]
    cfs_flipped = cfs[predictions == 1]
    recourse_cost_sum = dist_func_dataframe(factuals_flipped, cfs_flipped).sum()

    return n_flipped, recourse_cost_sum, mean_action


def cluster_results(
    model: Any,
    instances: pd.DataFrame,
    clusters: Dict[int, pd.DataFrame],
    cluster_expl_actions: Dict[int, pd.DataFrame],
    dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    numerical_features_names: List[str],
    categorical_features_names: List[str],
    cluster_action_choice_algo: Literal["max-eff", "mean-act", "low-cost", "min-cost-eff-thres", "eff-thres-hybrid"] = "max-eff",
    action_threshold: int = 2,
    num_low_cost: int = 20,
    effectiveness_threshold: float = 0.1,
    num_min_cost: Optional[int] = None,
    max_n_actions_full_combinations: int = 50,
) -> Tuple[Dict[int, Dict[str, Any]], float, float]:
    n_flipped_total = 0
    total_recourse_cost_sum = 0
    ret_clusters = {}
    for i, cluster in clusters.items():
        if cluster_action_choice_algo == "max-eff":
            n_flipped, recourse_cost_sum, selected_action = _select_action_max_eff(
                model=model,
                instances=cluster,
                candidate_actions=cluster_expl_actions[i],
                dist_func_dataframe=dist_func_dataframe,
                numerical_features_names=numerical_features_names,
                categorical_features_names=categorical_features_names,
            )
        elif cluster_action_choice_algo == "mean-act":
            n_flipped, recourse_cost_sum, selected_action = _select_action_mean(
                model=model,
                instances=cluster,
                candidate_actions=cluster_expl_actions[i],
                dist_func_dataframe=dist_func_dataframe,
                numerical_features_names=numerical_features_names,
                categorical_features_names=categorical_features_names,
            )
        elif cluster_action_choice_algo == "low-cost":
            n_flipped, recourse_cost_sum, selected_action = _select_action_low_cost(
                model=model,
                instances=instances,
                cluster_instances=cluster,
                candidate_actions=cluster_expl_actions[i],
                dist_func_dataframe=dist_func_dataframe,
                numerical_features_names=numerical_features_names,
                categorical_features_names=categorical_features_names,
                action_threshold=action_threshold,
                num_low_cost=num_low_cost,
                inv_total_clusters=(1 / len(clusters)),
            )
        # elif cluster_action_choice_algo == "min-cost-eff-thres":
        #     n_flipped, recourse_cost_sum, selected_action = _select_action_min_cost_eff_thres(
        #         model=model,
        #         instances=instances,
        #         cluster_instances=cluster,
        #         candidate_actions=cluster_expl_actions[i],
        #         dist_func_dataframe=dist_func_dataframe,
        #         numerical_features_names=numerical_features_names,
        #         categorical_features_names=categorical_features_names,
        #         effectiveness_threshold=effectiveness_threshold,
        #     )
        elif cluster_action_choice_algo == "min-cost-eff-thres-combinations":
            break
        elif cluster_action_choice_algo == "eff-thres-hybrid":
            break
        else:
            raise ValueError(
                "Unsupported algorithm for choice of final action for each cluster"
            )

        ret_clusters[i] = {
            "action": selected_action,
            "effectiveness": n_flipped / cluster.shape[0],
            "cost": recourse_cost_sum / n_flipped,
        }
        n_flipped_total += n_flipped
        total_recourse_cost_sum += recourse_cost_sum

    
    if cluster_action_choice_algo == "min-cost-eff-thres-combinations":
        n_flipped_total, total_recourse_cost_sum, action_set = _select_action_min_cost_eff_thres_combinations(
            model=model,
            instances=instances,
            clusters=clusters,
            candidate_actions=cluster_expl_actions,
            dist_func_dataframe=dist_func_dataframe,
            numerical_features_names=numerical_features_names,
            categorical_features_names=categorical_features_names,
            effectiveness_threshold=effectiveness_threshold,
            num_min_cost=num_min_cost,
        )
        
        assert len(action_set) == len(clusters)
        actions_iter = iter(action_set)
        ret_clusters = {i: {
            "action": next(actions_iter),
            "effectiveness": np.nan,
            "cost": np.nan,
        } for i in clusters.keys()}
        
        n_individuals_total = instances.shape[0]
        total_effectiveness_percentage = n_flipped_total / n_individuals_total
        total_mean_recourse_cost = total_recourse_cost_sum / n_flipped_total
        
        return ret_clusters, total_effectiveness_percentage, total_mean_recourse_cost
    # elif cluster_action_choice_algo == "eff-thres-hybrid":
    #     n_flipped_total, total_recourse_cost_sum, action_set = _select_actions_eff_thres_hybrid(
    #         model=model,
    #         instances=instances,
    #         clusters=clusters,
    #         candidate_actions=cluster_expl_actions,
    #         dist_func_dataframe=dist_func_dataframe,
    #         numerical_features_names=numerical_features_names,
    #         categorical_features_names=categorical_features_names,
    #         effectiveness_threshold=effectiveness_threshold,
    #         max_n_actions_full_combinations=max_n_actions_full_combinations,
    #     )
        
        assert len(action_set) == len(clusters)
        actions_iter = iter(action_set)
        ret_clusters = {i: {
            "action": next(actions_iter),
            "effectiveness": np.nan,
            "cost": np.nan,
        } for i in clusters.keys()}
        
        n_individuals_total = instances.shape[0]
        total_effectiveness_percentage = n_flipped_total / n_individuals_total
        total_mean_recourse_cost = total_recourse_cost_sum / n_flipped_total
        
        return ret_clusters, total_effectiveness_percentage, total_mean_recourse_cost
    else:
        n_individuals_total = sum(cluster.shape[0] for cluster in clusters.values())

        total_effectiveness_percentage = n_flipped_total / n_individuals_total
        total_mean_recourse_cost = total_recourse_cost_sum / n_flipped_total
        return ret_clusters, total_effectiveness_percentage, total_mean_recourse_cost


def print_results(
    clusters_stats: Dict[int, Dict[str, numbers.Number]],
    total_effectiveness: float,
    total_cost: float,
):
    for i, stats in enumerate(clusters_stats.values()):
        print(f"CLUSTER {i + 1} with size {stats['size']}:")
        display(pd.DataFrame(stats["action"]).T)
        print(f"Effectiveness: {stats['effectiveness']:.2%}, Cost: {stats['cost']:.2f}")

def format_glance_output(
    cluster_stats: Dict[int, Dict[str, numbers.Number]],
    categorical_columns: List[str],
):
    cluster_res = pd.DataFrame(cluster_stats)
    for index,row in cluster_res.T.reset_index(drop=True).iterrows():
    #     print(f"{Style.BRIGHT}CLUSTER {index+1}{Style.RESET_ALL} with size {row['size']}")
        output_string = f"{Style.BRIGHT}Action {index+1} \n{Style.RESET_ALL}"
        for column_name, value in row['action'].to_frame().T.reset_index(drop=True).iteritems():
            if column_name in categorical_columns:
                if value[0] != '-':
                    output_string += f"{Style.BRIGHT}{column_name}{Style.RESET_ALL} = {Fore.RED}{value[0]}{Fore.RESET} \n"
            else:
                if value[0] != '-':
                    if value[0] > 0 :
                        output_string += f"{Style.BRIGHT}{column_name}{Style.RESET_ALL} +{Fore.RED}{value[0]}{Fore.RESET} \n"
                    elif value[0] < 0 :
                        output_string += f"{Style.BRIGHT}{column_name}{Style.RESET_ALL} {Fore.RED}{value[0]}{Fore.RESET} \n"
        print(output_string)
        print(f"{Style.BRIGHT}Effectiveness:{Style.RESET_ALL} {Fore.GREEN}{row['effectiveness']:.2%}{Fore.RESET}\t{Style.BRIGHT}Cost:{Style.RESET_ALL} {Fore.MAGENTA}{row['cost']:.2f}{Fore.RESET}")
        print("\n")
        
def _merge_clusters(
    cluster1: int,
    cluster2: int,
    clusters: Dict[int, pd.DataFrame],
    cluster_explanations: Dict[int, pd.DataFrame],
    cluster_centroids: Dict[int, pd.DataFrame],
    cluster_expl_actions: Dict[int, pd.DataFrame],
    explanations_centroid: Dict[int, pd.DataFrame],
    numerical_features_names: List[str],
    categorical_features_names: List[str],
):
    clusters[cluster2] = pd.concat(
        [clusters[cluster2], clusters[cluster1]], ignore_index=True
    )
    del clusters[cluster1]

    cluster_explanations[cluster2] = pd.concat(
        [cluster_explanations[cluster2], cluster_explanations[cluster1]],
        ignore_index=True,
    )
    del cluster_explanations[cluster1]

    explanations_centroid[cluster2] = centroid_pandas(
        cluster_explanations[cluster2],
        numerical_columns=numerical_features_names,
        categorical_columns=categorical_features_names,
    )
    del explanations_centroid[cluster1]

    cluster_expl_actions[cluster2] = pd.concat(
        [cluster_expl_actions[cluster2], cluster_expl_actions[cluster1]],
        ignore_index=True,
    )
    del cluster_expl_actions[cluster1]

    cluster_centroids[cluster2] = centroid_pandas(
        clusters[cluster2],
        numerical_columns=numerical_features_names,
        categorical_columns=categorical_features_names,
    )
    del cluster_centroids[cluster1]


def _find_candidate_clusters(
    clusters: Dict[int, pd.DataFrame],
    cluster_centroids: Dict[int, pd.DataFrame],
    explanations_centroid: Dict[int, pd.DataFrame],
    heuristic_weights: Tuple[float, float],
    dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
) -> Tuple[int, int]:
    clusters_idx = clusters.keys()

    smallest_cluster = min(clusters_idx, key=lambda i: (clusters[i].shape[0], i))
    smallest_expl_centroid_repeat = pd.concat(
        [explanations_centroid[smallest_cluster]] * (len(clusters) - 1),
        ignore_index=True,
    )
    expl_centroids_rest = pd.concat(
        [explanations_centroid[i] for i in clusters_idx if i != smallest_cluster],
        ignore_index=True,
    )
    explanations_centroid_distances = dist_func_dataframe(
        smallest_expl_centroid_repeat,
        expl_centroids_rest,
    )
    smallest_centroid_repeat = pd.concat(
        [cluster_centroids[smallest_cluster]] * (len(clusters) - 1), ignore_index=True
    )
    centroids_rest = pd.concat(
        [cluster_centroids[i] for i in clusters_idx if i != smallest_cluster],
        ignore_index=True,
    )
    cluster_centroids_distances = dist_func_dataframe(
        smallest_centroid_repeat,
        centroids_rest,
    )
    merge_heuristic_values = (
        heuristic_weights[0] * cluster_centroids_distances
        + heuristic_weights[1] * explanations_centroid_distances
    )
    candidates = [
        (smallest_cluster, cluster1)
        for cluster1 in clusters_idx
        if cluster1 != smallest_cluster
    ]
    candidates = [
        (c1, c2, merge_heuristic_values.iloc[i])
        for i, (c1, c2) in enumerate(candidates)
    ]

    candidates.sort(key=lambda x: (x[2], x[1]))

    return candidates[0][0], candidates[0][1]


def _generate_clusters(
    instances: pd.DataFrame,
    num_clusters: int,
    categorical_features_names: List[str],
    clustering_method: ClusteringMethod,
) -> Dict[int, pd.DataFrame]:
    ohe_instances = _one_hot_encode(instances, categorical_features_names)
    clustering_method.fit(ohe_instances)
    assigned_clusters = clustering_method.predict(ohe_instances)

    cluster_ids = np.unique(assigned_clusters)
    cluster_ids.sort()
    clusters = {i: instances.iloc[assigned_clusters == i] for i in cluster_ids}

    return clusters


def _one_hot_encode(X: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    transformer = ColumnTransformer(
        [("ohe", OneHotEncoder(sparse_output=False), categorical_columns)],
        remainder="passthrough",
    )
    ret = transformer.fit_transform(X)
    assert isinstance(ret, np.ndarray)
    return pd.DataFrame(ret, columns=transformer.get_feature_names_out())
