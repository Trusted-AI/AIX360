from typing import Union, Any, List, Optional, Dict, Tuple, Callable
from ..base import GlobalCounterfactualMethod, LocalCounterfactualMethod
from ..iterative_merges.iterative_merges import C_GLANCE, _select_action_max_eff
import pandas as pd
from ..utils.metadata_requests import _decide_local_cf_method
from ..utils.centroid import centroid_pandas
from ..utils.action import extract_actions_pandas, apply_action_pandas
from sklearn.inspection import permutation_importance
from ..iterative_merges.iterative_merges import cumulative
from ..counterfactual_costs import build_dist_func_dataframe
from .node import Node
import numpy as np
from tqdm import tqdm


class T_GLANCE:
    """
    A class to generate counterfactual explanations using a decision tree-like structure.
    
    This class allows users to create a tree structure for counterfactual generation,
    optimizing effectiveness and cost based on specified features. It supports both local 
    and global methods for generating counterfactuals.

    Attributes:
    ----------
    model : Any
        The predictive model used for generating counterfactuals.
    split_features : Union[List, int]
        Features to split the tree. Can be a list of feature names or an integer specifying
        the number of top features to use based on permutation importance.
    partition_counterfactuals : int
        The number of partitions to create for counterfactuals.
    child_count : int
        The number of children each node can have.
    global_method : Union[GlobalCounterfactualMethod, str]
        The global counterfactual generation method to use.
    local_method : Union[LocalCounterfactualMethod, str]
        The local counterfactual generation method to use.
    num_local_counterfactuals : int
        The number of local counterfactuals to generate.
    node : Node
        The root node of the counterfactual tree.
    node_instances : pd.DataFrame
        The instances that were used to build the counterfactual tree.
    dist_func_dataframe : Callable
        A distance function for calculating distances between instances.

    Methods:
    -------
    fit(X, y, train_dataset=None, feat_to_vary="all", random_seed=13, numeric_features_names=None, categorical_features_names=None):
        Fits the counterfactual tree to the provided data.
    
    _local_group_eff_cost(instances):
        Calculates the effectiveness and cost of local counterfactuals for a group of instances.
    
    _group_eff_cost(instances):
        Calculates the effectiveness and cost of counterfactuals for a group of instances, 
        utilizing local or global methods.
    
    partition_group(instances):
        Partitions the group of instances into a tree structure based on the specified features.
    
    cumulative_leaf_actions():
        Computes the total effectiveness and cost of actions taken from leaf nodes of the tree.
    
    """

    def __init__(
        self,
        model: Any,
        split_features: Union[List, int] = None,
        partition_counterfactuals: int = None,
        child_count: int = 2,
        global_method: Union[GlobalCounterfactualMethod, str] = None,
        local_method: Union[LocalCounterfactualMethod, str] = None,
        num_local_counterfactuals: int = 100,
    ):
        """
        Initializes the CounterfactualTree instance.

        Parameters:
        ----------
        model : Any
            The predictive model to use for generating counterfactuals.
        split_features : Union[List, int], optional
            Features to split the tree. If None, uses permutation importance to select.
            If an integer, selects the top N features.
        partition_counterfactuals : int, optional
            Number of partitions for counterfactual generation.
        child_count : int, optional
            Number of children for each node in the tree. Default is 2.
        global_method : Union[GlobalCounterfactualMethod, str], optional
            The global counterfactual generation method to use.
        local_method : Union[LocalCounterfactualMethod, str], optional
            The local counterfactual generation method to use.
        num_local_counterfactuals : int, optional
            Number of local counterfactuals to generate. Default is 100.
        """
        self.model = model
        self.split_features = split_features
        self.partition_counterfactuals = partition_counterfactuals
        self.child_count = child_count
        self.global_method = global_method
        self.local_method = local_method
        self.num_local_counterfactuals = num_local_counterfactuals

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_dataset: Optional[pd.DataFrame] = None,
        feat_to_vary: Optional[Union[List[str], str]] = "all",
        random_seed: int = 13,
        numeric_features_names: Optional[List[str]] = None,
        categorical_features_names: Optional[List[str]] = None,
    ):
        """
        Fits the counterfactual tree to the provided data.

        Parameters:
        ----------
        X : pd.DataFrame
            Features of the dataset.
        y : pd.Series
            Target variable.
        train_dataset : Optional[pd.DataFrame], optional
            The training dataset to use for local counterfactual generation methods.
        feat_to_vary : Optional[Union[List[str], str]], optional
            Features to vary in counterfactual generation. Default is "all".
        random_seed : int, optional
            Random seed for reproducibility. Default is 13.
        numeric_features_names : Optional[List[str]], optional
            List of numeric feature names. If None, they will be inferred from X.
        categorical_features_names : Optional[List[str]], optional
            List of categorical feature names. If None, they will be inferred from X.
        """
        if self.split_features == None:
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=30, random_state=42
            )

            feature_names = X.columns

            mean_importance = perm_importance.importances_mean
            top_indices = mean_importance.argsort()[-2:][::-1]
            top_features = feature_names[top_indices]

            self.split_features = list(top_features)
        elif isinstance(self.split_features, int):
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=30, random_state=42
            )
            
            feature_names = X.columns

            mean_importance = perm_importance.importances_mean
            top_indices = mean_importance.argsort()[-self.split_features:][::-1]
            top_features = feature_names[top_indices]

            self.split_features = list(top_features)

        self.split_values = _get_split_values(X, self.split_features, self.child_count)

        if numeric_features_names is None:
            if categorical_features_names is None:
                numeric_features_names = X.select_dtypes(
                    include=["number"]
                ).columns.tolist()
            else:
                numeric_features_names = X.columns.difference(
                    categorical_features_names
                ).tolist()
        if categorical_features_names is None:
            categorical_features_names = X.columns.difference(
                numeric_features_names
            ).tolist()

        self.numerical_features_names = numeric_features_names
        self.categorical_features_names = categorical_features_names
        self.X = X
        self.y = y
        self.train_dataset = train_dataset
        self.random_seed = random_seed
        self.feat_to_vary = feat_to_vary
        self.dist_func_dataframe = build_dist_func_dataframe(
            self.X, self.numerical_features_names, self.categorical_features_names
        )

        if self.local_method == None:
            backup = "Dice"
        else:
            backup = self.local_method
        self.cf_generator_backup = _decide_local_cf_method(
            method=backup,
            model=self.model,
            train_dataset=self.train_dataset,
            numeric_features_names=self.numerical_features_names,
            categorical_features_names=self.categorical_features_names,
            feat_to_vary=self.feat_to_vary,
            random_seed=random_seed,
        )

        if self.global_method == None and self.local_method == None:
            self.generation_method = "Global-IM"
            if self.partition_counterfactuals == None:
                self.partition_counterfactuals = 3
            self.cf_generator = C_GLANCE(
                self.model, final_clusters=self.partition_counterfactuals, verbose=False
            )
            if self.train_dataset is None:
                raise ValueError(
                    "You need to pass train_dataset for Dice if you want default Iterative merges."
                )
            self.cf_generator.fit(X, y, self.train_dataset)
        elif self.global_method != None:
            self.generation_method = "Global"
            if self.partition_counterfactuals == None:
                self.partition_counterfactuals = 3
            self.cf_generator = self.global_method
        else:
            self.generation_method = "Local"
            if self.partition_counterfactuals == None:
                self.partition_counterfactuals = 1

    def _local_group_eff_cost(self, instances):
        """
        Calculates the effectiveness and cost of local counterfactuals for a group of instances.

        Parameters:
        ----------
        instances : pd.DataFrame
            The group of instances to analyze.

        Returns:
        -------
        Tuple[float, float, List[Any]]
            A tuple containing the effectiveness, cost, and list of actions.
        """
        centroid = centroid_pandas(
            instances,
            self.numerical_features_names,
            self.categorical_features_names,
        )
        cfs = self.cf_generator_backup.explain_instances(
            centroid,
            self.num_local_counterfactuals,
        )
        if cfs.shape[0] == 0:
            return 0, 0, []
        
        actions = extract_actions_pandas(
            X=pd.concat([centroid] * cfs.shape[0]).set_index(
                cfs.index
            ),
            cfs=cfs,
            categorical_features=self.categorical_features_names,
            numerical_features=self.numerical_features_names,
            categorical_no_action_token="-",
        )
        # actions = [action for _, action in actions.iterrows()]
        actions_info = _select_action_max_eff(
            self.model,
            instances,
            actions,
            self.dist_func_dataframe,
            self.numerical_features_names,
            self.categorical_features_names,
            self.partition_counterfactuals,
        )
        if type(actions_info) is not list:
            actions_info = [actions_info]
        actions = [action for _, _, action in actions_info]
        eff, cost = cumulative(
            self.model,
            instances,
            actions,
            self.dist_func_dataframe,
            self.numerical_features_names,
            self.categorical_features_names,
            "-",
        )
        return eff, cost, actions

    def _group_eff_cost(
        self,
        instances,
    ):
        """
        Calculates the effectiveness and cost of counterfactuals for a group of instances,
        utilizing local or global methods.

        Parameters:
        ----------
        instances : pd.DataFrame
            The group of instances to analyze.

        Returns:
        -------
        Tuple[float, float, List[Any]]
            A tuple containing the effectiveness, cost, and list of actions.
        """

        if self.generation_method == "Local":
            return self._local_group_eff_cost(instances)
        elif self.generation_method == "Global-IM":
            clusters = min(100, len(instances))
            if clusters < self.partition_counterfactuals:
                return self._local_group_eff_cost(instances)
            else:
                self.cf_generator.initial_clusters = clusters
                eff, cost = self.cf_generator.explain_group(instances)
                actions = self.cf_generator.global_actions()
        elif self.generation_method == 'Global':
            eff, cost = self.cf_generator.explain_group(instances)
            actions = self.cf_generator.global_actions()
        else:
            raise ValueError("Generation method does not exist")
            

        return eff, cost, actions

    def partition_group(self, instances: pd.DataFrame):
        """
        Partitions the group of instances into a tree structure based on the specified features.

        Parameters:
        ----------
        instances : pd.DataFrame
            The group of instances to partition.

        Returns:
        -------
        Node
            The root node of the partitioned tree.
        """

        def _partition_group(
            group, split_features, eff_prec=None, cost_prec=None, actions_prec=None
        ):

            if eff_prec == None:
                eff_node, cost_node, actions = self._group_eff_cost(group)
            else:
                eff_node, cost_node, actions = eff_prec, cost_prec, actions_prec

            node = Node(
                effectiveness=eff_node, cost=cost_node, actions=actions, size=len(group)
            )
            possible_splits = []

            for feature in split_features:
                eff_children, cost_children = 0, 0
                children_info = []

                for feature_split_values in self.split_values[feature]:
                    split_df = group[group[feature].isin(feature_split_values)]

                    if not split_df.empty:
                        eff_child, cost_child, actions = self._group_eff_cost(split_df)
                        eff_children += eff_child
                        cost_children += cost_child
                        children_info.append(
                            (
                                feature_split_values,
                                split_df,
                                eff_child,
                                cost_child,
                                actions,
                            )
                        )

                possible_splits.append(
                    (feature, eff_children, cost_children, children_info)
                )

            if len(possible_splits) == 0:
                return node

            possible_splits = sorted(possible_splits, key=lambda x: -x[1])
            node.split_feature = possible_splits[0][0]
            split_features.remove(node.split_feature)
            child_info = possible_splits[0][3]

            for child in child_info:
                child_node = _partition_group(
                    child[1], split_features, child[2], child[3], child[4]
                )

                node.add_child(child[0], child_node)

            return node

        self.node = _partition_group(instances, self.split_features)
        self.node_instances = instances
        return self.node

    def cumulative_leaf_actions(self):
        """
        Computes the total effectiveness and cost of actions taken from leaf nodes of the tree.

        Returns:
        -------
        Tuple[float, float, int]
            A tuple containing the total effectiveness, total cost, and the number of actions taken.
        """
        eff, cost = cumulative(
            self.model,
            self.node_instances,
            self.node.return_leafs_actions(),
            self.dist_func_dataframe,
            self.numerical_features_names,
            self.categorical_features_names,
            categorical_no_action_token="-",
        )

        print(f"\nTOTAL EFFECTIVENESS: {eff / self.node_instances.shape[0]:.2%}")
        print(f"\nTOTAL COST: {(cost / eff if eff > 0 else 0):.2f}")

        return eff, cost, len(self.node.return_leafs_actions())


def _split_list(lst, n):
    """
    Splits a list into n approximately equal parts.

    Parameters:
    ----------
    lst : list
        The list to split.
    n : int
        The number of parts to split the list into.

    Returns:
    -------
    Generator
        A generator yielding the split parts of the list.
    """
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def _get_split_values(X, split_features, child_count):
    """
    Generates split values for the specified features.

    Parameters:
    ----------
    X : pd.DataFrame
        The dataset containing features.
    split_features : List[str]
        The list of features to create split values for.
    child_count : int
        The number of child splits to create for each feature.

    Returns:
    -------
    Dict[str, List[List[Any]]]
        A dictionary mapping each feature to its corresponding split values.
    """
    split_values = {}
    for feature in split_features:
        lst = sorted(list(X[feature].unique()))
        split_count = child_count
        if child_count == -1:
            split_count = len(lst)
        split_values[feature] = list(_split_list(lst, split_count))
    return split_values
