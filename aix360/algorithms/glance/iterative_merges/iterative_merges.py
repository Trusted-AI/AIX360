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


class C_GLANCE(GlobalCounterfactualMethod):
    """
    A class for generating global counterfactual explanations using an iterative merging approach.
    
    It allows the user to control the number of clusters and the methods used 
    for clustering and generating counterfactuals.

    Attributes:
    ----------
    model : Any
        The predictive model used for generating counterfactuals.
    initial_clusters : int
        The initial number of clusters to form.
    final_clusters : int
        The target number of clusters after merging.
    num_local_counterfactuals : int
        The number of local counterfactuals to generate for each cluster.
    heuristic_weights : Tuple[float, float]
        Weights used in the heuristic for merging clusters.
    alternative_merges : bool
        If True, allows alternative merging strategies.
    random_seed : int
        Seed for random number generation.
    verbose : bool
        If True, enables verbose output during processing.
    final_clustering : Optional[Dict[int, pd.DataFrame]]
        The final clustering of instances after merging.
    cluster_results : Optional[Dict[int, Dict[str, Any]]]
        Results of the clustering including effectiveness and cost metrics.

    Methods:
    -------
    _set_features_names(X, numerical_names, categorical_names):
        Sets the feature names for numerical and categorical features.
    
    fit(X, y, train_dataset, feat_to_vary, numeric_features_names, categorical_features_names,
        clustering_method, cf_generator, cluster_action_choice_algo, ...)
        Fits the clustering and counterfactual generation model to the provided dataset.
    
    explain_group(instances):
        Explains the group of instances by generating counterfactuals based on clustering.
    
    global_actions():
        Retrieves the global actions derived from the clustered results.
    """

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
        """
        Initializes the IterativeMerges instance.

        Parameters:
        ----------
        model : Any
            The predictive model used for generating counterfactuals.
        initial_clusters : int, optional
            The initial number of clusters to form. Default is 100.
        final_clusters : int, optional
            The target number of clusters after merging. Default is 10.
        num_local_counterfactuals : int, optional
            The number of local counterfactuals to generate for each cluster. Default is 5.
        heuristic_weights : Tuple[float, float], optional
            Weights used in the heuristic for merging clusters. Default is (0.5, 0.5).
        alternative_merges : bool, optional
            If True, allows alternative merging strategies. Default is True.
        random_seed : int, optional
            Seed for random number generation. Default is 13.
        verbose : bool, optional
            If True, enables verbose output during processing. Default is True.
        """
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
        """
        Sets the feature names for numerical and categorical features.

        Parameters:
        ----------
        X : pd.DataFrame
            The dataset to analyze.
        numerical_names : Optional[List[str]]
            List of numerical feature names. If None, they will be inferred from X.
        categorical_names : Optional[List[str]]
            List of categorical feature names. If None, they will be inferred from X.

        Returns:
        -------
        Tuple[List[str], List[str]]
            A tuple containing lists of numerical and categorical feature names.
        """
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
        cluster_action_choice_algo: Literal["max-eff", "mean-act", "low-cost"] = "max-eff",
        nns__n_scalars: Optional[int] = None,
        rs__n_most_important: Optional[int] = None,
        rs__n_categorical_most_frequent: Optional[int] = None,
        lowcost__action_threshold: Optional[int] = None,
        lowcost__num_low_cost: Optional[int] = None,
        min_cost_eff_thres__effectiveness_threshold: Optional[float] = None,
        min_cost_eff_thres_combinations__num_min_cost: Optional[int] = None,
        eff_thres_hybrid__max_n_actions_full_combinations: Optional[int] = None,
    ) -> "C_GLANCE":
        """
        Fits the clustering and counterfactual generation model to the provided dataset.

        Parameters:
        ----------
        X : pd.DataFrame
            Features of the dataset.
        y : pd.Series
            Target variable.
        train_dataset : pd.DataFrame
            The training dataset used for local counterfactual generation methods.
        feat_to_vary : Optional[Union[List[str], str]], optional
            Features to vary in counterfactual generation. Default is "all".
        numeric_features_names : Optional[List[str]], optional
            List of numeric feature names. If None, they will be inferred from X.
        categorical_features_names : Optional[List[str]], optional
            List of categorical feature names. If None, they will be inferred from X.
        clustering_method : Union[ClusteringMethod, Literal["KMeans"]], optional
            The clustering method to use. Default is "KMeans".
        cf_generator : Union[LocalCounterfactualMethod, Literal["Dice", "NearestNeighbors", "RandomSampling"]], optional
            The local counterfactual generation method to use. Default is "Dice".
        cluster_action_choice_algo : Literal["max-eff", "mean-act", "low-cost""], optional
            The algorithm for selecting actions from clusters. Default is "max-eff".
        nns__n_scalars : Optional[int], optional
            Number of scalar features to use for nearest neighbors. Default is None.
        rs__n_most_important : Optional[int], optional
            Number of most important features for random sampling. Default is None.
        rs__n_categorical_most_frequent : Optional[int], optional
            Number of most frequent categorical features for random sampling. Default is None.
        lowcost__action_threshold : Optional[int], optional
            Action threshold for low-cost methods. Default is None.
        lowcost__num_low_cost : Optional[int], optional
            Number of low-cost actions to consider. Default is None.
        min_cost_eff_thres__effectiveness_threshold : Optional[float], optional
            Effectiveness threshold for minimum cost methods. Default is None.
        min_cost_eff_thres_combinations__num_min_cost : Optional[int], optional
            Number of minimum cost combinations to evaluate. Default is None.
        eff_thres_hybrid__max_n_actions_full_combinations : Optional[int], optional
            Maximum number of actions for full combinations in hybrid thresholding. Default is None.

        Returns:
        -------
        IterativeMerges
            Returns the fitted instance of IterativeMerges.
        """
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
        """
        Explains the group of instances by generating counterfactuals based on clustering.

        Parameters:
        ----------
        instances : pd.DataFrame
            The group of instances to explain.

        Returns:
        -------
        Tuple[int, float]
            A tuple containing the total effectiveness and total cost of the generated counterfactuals.
        """
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
    """
    Computes the cumulative effectiveness and cost of applying a set of actions 
    to a given set of instances using a predictive model.

    Parameters:
    ----------
    model : Any
        A predictive model with a predict method. This model will be used to predict 
        outcomes after applying actions to the input instances.
    instances : pd.DataFrame
        A DataFrame containing the instances for which actions are to be applied.
    actions : List[dict]
        A list of actions, where each action is represented as a dictionary that 
        specifies how to modify the instances.
    dist_func_dataframe : Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
        A distance function that takes two DataFrames and returns a Series of distances 
        between corresponding rows.
    numeric_features_names : List[str]
        A list of names for the numeric features in the instances DataFrame.
    categorical_features_names : List[str]
        A list of names for the categorical features in the instances DataFrame.
    categorical_no_action_token : Any
        A token used to represent a no-action state for categorical features.

    Returns:
    -------
    Tuple[int, float]
        A tuple containing:
        - effectiveness: An integer count of how many actions were effective (i.e., 
          resulted in a finite cost).
        - cost: A float representing the total cost incurred by the effective actions. 
    """
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
    """
    Selects the action with the lowest cost that flips a sufficient number of instances 
    in the given dataset, based on a predictive model.

    This function evaluates candidate actions, applies them to the provided instances, 
    and calculates the number of predictions that were flipped as a result. It returns 
    the action that results in the lowest recourse cost while also meeting a specified 
    threshold of flipped predictions.

    Parameters:
    ----------
    model : Any
        A machine learning model used for making predictions.

    instances : pd.DataFrame
        A DataFrame containing the instances for which counterfactuals are being generated.

    cluster_instances : pd.DataFrame
        A DataFrame containing instances from a specific cluster used for evaluating actions.

    candidate_actions : pd.DataFrame
        A DataFrame containing potential actions to apply to the instances.

    dist_func_dataframe : Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
        A function that computes the distance or cost between two DataFrames.

    numerical_features_names : List[str]
        A list of names for the numerical features in the instances.

    categorical_features_names : List[str]
        A list of names for the categorical features in the instances.

    action_threshold : int
        The minimum ratio of flipped predictions to total instances required to consider 
        an action effective.

    num_low_cost : int
        The maximum number of low-cost actions to evaluate.

    inv_total_clusters : int
        The inverse of the total number of clusters used for normalization.

    Returns:
    -------
    Tuple[int, float, pd.Series]
        A tuple containing:
        - The number of predictions flipped.
        - The minimum recourse cost associated with the best action.
        - The best action selected from the candidate actions.

    Raises:
    ------
    ValueError
        If no actions are found that meet the effectiveness threshold.
    """
    actions_list = [action for _, action in candidate_actions.iterrows()]
    actions_list.sort(
        key=lambda action: action_fake_cost(
            action, numerical_features_names, categorical_features_names
        )
    )
    cf_list = []
    for action in actions_list[: min(num_low_cost, len(actions_list))]:
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

def actions_cumulative_eff_cost(
    model: Any,
    X: pd.DataFrame,
    actions_with_costs: List[Tuple[pd.Series, float]],
    dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    numerical_columns: List[str],
    categorical_columns: List[str],
    categorical_no_action_token: Any,
) -> Tuple[float, float]:
    """
    Evaluates the cumulative effectiveness and cost of applying a sequence of actions 
    to a dataset using a predictive model.

    This function applies each action from the sorted list of actions with their costs, 
    predicts the outcomes, and calculates the total number of predictions that were flipped 
    as well as the total recourse cost incurred from the actions.

    Parameters:
    ----------
    model : Any
        A machine learning model used for making predictions on the modified instances.

    X : pd.DataFrame
        The original DataFrame of instances to which actions will be applied.

    actions_with_costs : List[Tuple[pd.Series, float]]
        A list of tuples where each tuple contains:
        - A pandas Series representing the action to apply.
        - A float representing the cost associated with the action.

    dist_func_dataframe : Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
        A function that computes the distance or cost between two DataFrames.

    numerical_columns : List[str]
        A list of names for the numerical columns in the DataFrame.

    categorical_columns : List[str]
        A list of names for the categorical columns in the DataFrame.

    categorical_no_action_token : Any
        A token used to represent the absence of an action for categorical features.

    Returns:
    -------
    Tuple[float, float]
        A tuple containing:
        - The total number of predictions flipped across all actions applied.
        - The total recourse cost incurred from applying the actions.
    """
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

def _select_action_max_eff(
    model: Any,
    instances: pd.DataFrame,
    candidate_actions: pd.DataFrame,
    dist_func_dataframe: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    numerical_features_names: List[str],
    categorical_features_names: List[str],
    num_actions: int = 1,
) -> Tuple[int, int, pd.Series]:
    """
    Selects actions based on maximizing the effectiveness.

    This function evaluates a set of candidate actions by applying each action to the given
    instances, predicting the outcomes, and calculating the number of predictions that are
    flipped (changed from 0 to 1). It also computes the recourse cost associated with each action.
    Depending on the number of actions specified, it returns either the best action or a list
    of the top actions based on effectiveness.

    Parameters:
    ----------
    model : Any
        A machine learning model used for making predictions on the modified instances.

    instances : pd.DataFrame
        The DataFrame of original instances to which actions will be applied.

    candidate_actions : pd.DataFrame
        A DataFrame containing the candidate actions to evaluate.

    dist_func_dataframe : Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
        A function that computes the distance or cost between two DataFrames.

    numerical_features_names : List[str]
        A list of names for the numerical columns in the DataFrame.

    categorical_features_names : List[str]
        A list of names for the categorical columns in the DataFrame.

    num_actions : int, optional
        The number of top actions to select based on effectiveness. Defaults to 1.

    Returns:
    -------
    Tuple[int, int, pd.Series]
        If `num_actions` is 1, returns:
        - The maximum number of predictions flipped.
        - The total recourse cost associated with the best action.
        - The best action (pd.Series).
        
        If `num_actions` > 1, returns a list of the top actions based on their effectiveness.
    """
    max_n_flipped = 0
    cf_list = []

    for _, action in candidate_actions.iterrows():
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
    """
    Selects the mean action from a set of candidate actions and evaluates its effectiveness.

    This function computes the mean action from the candidate actions and applies it to the
    given instances. It then predicts the outcomes and calculates the number of predictions that
    are flipped (changed from 0 to 1) as well as the associated recourse cost.

    Parameters:
    ----------
    model : Any
        A machine learning model used for making predictions on the modified instances.

    instances : pd.DataFrame
        The DataFrame of original instances to which the mean action will be applied.

    candidate_actions : pd.DataFrame
        A DataFrame containing the candidate actions from which the mean action will be derived.

    dist_func_dataframe : Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
        A function that computes the distance or cost between two DataFrames.

    numerical_features_names : List[str]
        A list of names for the numerical columns in the DataFrame.

    categorical_features_names : List[str]
        A list of names for the categorical columns in the DataFrame.

    Returns:
    -------
    Tuple[int, int, pd.Series]
        A tuple containing:
        - The number of predictions flipped by applying the mean action.
        - The total recourse cost associated with the mean action.
        - The mean action (pd.Series).
    """
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
    """
    Evaluates and selects actions for each cluster based on a specified action choice algorithm.

    This function iterates through each cluster of instances, applying the specified algorithm to 
    select the best action for achieving recourse while minimizing costs. It calculates the total 
    effectiveness and mean recourse costs across all clusters.

    Parameters:
    ----------
    model : Any
        A machine learning model used for making predictions on modified instances.

    instances : pd.DataFrame
        The DataFrame of original instances to which actions will be applied.

    clusters : Dict[int, pd.DataFrame]
        A dictionary mapping cluster IDs to DataFrames of instances belonging to each cluster.

    cluster_expl_actions : Dict[int, pd.DataFrame]
        A dictionary mapping cluster IDs to DataFrames of candidate actions for each cluster.

    dist_func_dataframe : Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
        A function that computes the distance or cost between two DataFrames.

    numerical_features_names : List[str]
        A list of names for the numerical columns in the DataFrames.

    categorical_features_names : List[str]
        A list of names for the categorical columns in the DataFrames.

    cluster_action_choice_algo : Literal["max-eff", "mean-act", "low-cost", "min-cost-eff-thres", "eff-thres-hybrid"]
        The algorithm to use for selecting actions from candidate actions. Options include:
        - "max-eff": Select the action with maximum effectiveness.
        - "mean-act": Select the mean action from candidate actions.
        - "low-cost": Select actions based on low cost.

    action_threshold : int
        Minimum threshold for the number of flipped predictions required to consider an action effective.

    num_low_cost : int
        The number of low-cost actions to consider (used when the low-cost algorithm is selected).

    effectiveness_threshold : float
        Minimum effectiveness required for actions (used when the min-cost-eff-thres algorithm is selected).

    num_min_cost : Optional[int]
        Number of minimum cost actions to consider (used when the min-cost-eff-thres algorithm is selected).

    max_n_actions_full_combinations : int
        Maximum number of actions to evaluate in full combinations (not currently used in the function).

    Returns:
    -------
    Tuple[Dict[int, Dict[str, Any]], float, float]
        A tuple containing:
        - A dictionary where each key is a cluster ID and each value is another dictionary with the selected action, its effectiveness, and cost.
        - Total effectiveness percentage across all clusters.
        - Total mean recourse cost across all clusters.
    """
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
    """
    Prints the statistics for each cluster, including effectiveness and cost.

    This function takes the results of cluster analysis and formats them for easy 
    viewing. It displays the size of each cluster, the actions taken, and the 
    effectiveness and cost of those actions.

    Parameters:
    ----------
    clusters_stats : Dict[int, Dict[str, numbers.Number]]
        A dictionary where keys are cluster IDs (integers) and values are 
        dictionaries containing statistics for each cluster. Each value dictionary
        must contain the following keys:
            - "size": The size of the cluster.
            - "action": The actions taken for the cluster.
            - "effectiveness": The effectiveness of the actions in the cluster.
            - "cost": The cost associated with the actions.

    total_effectiveness : float
        The total effectiveness percentage across all clusters, represented as a decimal 
        (e.g., 0.75 for 75%).

    total_cost : float
        The total cost associated with the actions taken across all clusters.
    """
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
    """
    Merges two clusters into one and updates all associated data structures.

    This function takes two cluster identifiers and combines their respective data.
    It updates the clusters, explanations, centroids, and action dataframes accordingly.

    Parameters:
    ----------
    cluster1 : int
        The identifier for the first cluster to merge.

    cluster2 : int
        The identifier for the second cluster to merge into.

    clusters : Dict[int, pd.DataFrame]
        A dictionary mapping cluster IDs to their respective dataframes.

    cluster_explanations : Dict[int, pd.DataFrame]
        A dictionary mapping cluster IDs to their explanations dataframes.

    cluster_centroids : Dict[int, pd.DataFrame]
        A dictionary mapping cluster IDs to their centroid dataframes.

    cluster_expl_actions : Dict[int, pd.DataFrame]
        A dictionary mapping cluster IDs to their explanation actions dataframes.

    explanations_centroid : Dict[int, pd.DataFrame]
        A dictionary mapping cluster IDs to their centroid explanations dataframes.

    numerical_features_names : List[str]
        A list of names for the numerical features in the dataset.

    categorical_features_names : List[str]
        A list of names for the categorical features in the dataset.
    """
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
    """
    Identifies the best candidate clusters for merging based on distances of centroids
    and explanation centroids, weighted by given heuristic values.

    The function selects the smallest cluster and calculates distances to all other clusters' centroids.
    It uses these distances to determine a heuristic value for potential merges, returning the two 
    clusters with the best merge heuristic.

    Parameters:
    ----------
    clusters : Dict[int, pd.DataFrame]
        A dictionary mapping cluster IDs to their respective dataframes.

    cluster_centroids : Dict[int, pd.DataFrame]
        A dictionary mapping cluster IDs to their centroid dataframes.

    explanations_centroid : Dict[int, pd.DataFrame]
        A dictionary mapping cluster IDs to their explanation centroids.

    heuristic_weights : Tuple[float, float]
        A tuple containing two weights used to combine centroid distances and explanation centroid distances.

    dist_func_dataframe : Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
        A function that computes the distance between two dataframes, returning a series of distances.

    Returns:
    -------
    Tuple[int, int]
        A tuple containing the IDs of the two candidate clusters identified for merging.
    """
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
    """
    Generates clusters from the given instances using the specified clustering method.

    The function applies one-hot encoding to the categorical features in the input data,
    fits the provided clustering method, and assigns instances to clusters. It returns 
    a dictionary mapping cluster IDs to their respective dataframes.

    Parameters:
    ----------
    instances : pd.DataFrame
        The input data containing instances to be clustered.

    num_clusters : int
        The desired number of clusters to generate. Note that the actual number of 
        clusters may vary depending on the clustering method used.

    categorical_features_names : List[str]
        A list of names of categorical features in the input data that need to be 
        one-hot encoded for clustering.

    clustering_method : ClusteringMethod
        An instance of a clustering method (e.g., KMeans, DBSCAN) that implements 
        the fit and predict methods.

    Returns:
    -------
    Dict[int, pd.DataFrame]
        A dictionary where the keys are cluster IDs and the values are dataframes 
        containing the instances assigned to each cluster.
    """
    ohe_instances = _one_hot_encode(instances, categorical_features_names)
    clustering_method.fit(ohe_instances)
    assigned_clusters = clustering_method.predict(ohe_instances)

    cluster_ids = np.unique(assigned_clusters)
    cluster_ids.sort()
    clusters = {i: instances.iloc[assigned_clusters == i] for i in cluster_ids}

    return clusters


def _one_hot_encode(X: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """
    Applies one-hot encoding to the specified categorical columns of a DataFrame.

    This function transforms categorical columns in the input DataFrame into 
    a one-hot encoded format, allowing them to be used in machine learning models. 
    The non-categorical columns are retained in their original form.

    Parameters:
    ----------
    X : pd.DataFrame
        The input DataFrame containing the data with both categorical and numerical features.

    categorical_columns : List[str]
        A list of names of the categorical columns in the DataFrame that should be one-hot encoded.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame where the specified categorical columns have been one-hot encoded, 
        and all other columns are retained as is.
    """
    transformer = ColumnTransformer(
        [("ohe", OneHotEncoder(sparse_output=False), categorical_columns)],
        remainder="passthrough",
    )
    ret = transformer.fit_transform(X)
    assert isinstance(ret, np.ndarray)
    return pd.DataFrame(ret, columns=transformer.get_feature_names_out())
