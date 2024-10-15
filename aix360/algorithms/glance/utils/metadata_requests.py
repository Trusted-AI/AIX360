from ..base import ClusteringMethod, LocalCounterfactualMethod
from ..clustering import KMeansMethod
from ..local_cfs import DiceMethod, NearestNeighborMethod, RandomSampling


def _decide_cluster_method(method, n_clusters, random_seed) -> ClusteringMethod:
    """
    Determines and returns the appropriate clustering method based on the input `method` argument.

    If `method` is a string specifying a known clustering algorithm, the function initializes the corresponding clustering method (e.g., KMeans). 
    If `method` is already an instance of a clustering method, it is returned unchanged.

    Parameters:
    ----------
    method : str or ClusteringMethod
        The desired clustering method. This can either be a string specifying a supported clustering method (e.g., "KMeans") or an instance of a clustering method.
    n_clusters : int
        The number of clusters to use in the clustering algorithm.
    random_seed : int
        A seed for the random number generator to ensure reproducibility.

    Returns:
    -------
    ClusteringMethod
        An instance of the appropriate clustering method based on the input. For example, if `method` is "KMeans", an instance of `KMeansMethod` is returned.

    Raises:
    -------
    ValueError
        If an unsupported string is passed as the `method` argument.
    """
    if isinstance(method, str):
        if method == "KMeans":
            method = KMeansMethod(num_clusters=n_clusters, random_seed=random_seed)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    else:
        method = method
    return method


def _decide_local_cf_method(
    method, model, train_dataset, numeric_features_names, 
    categorical_features_names, feat_to_vary, 
    random_seed, n_most_important: int = 15, 
    n_categorical_most_frequent: int = 15, 
    n_scalars: int = 1000,
) -> LocalCounterfactualMethod:
    """
    Determines and returns the appropriate local counterfactual method based on the input `method` argument.

    This function initializes the specified local counterfactual method (e.g., "Dice", "NearestNeighbors", or "RandomSampling") and fits it to the provided training dataset.
    If the `method` is already an instance of a local counterfactual method, it is returned unchanged.

    Parameters:
    ----------
    method : str or LocalCounterfactualMethod
        The desired local counterfactual method. This can either be a string specifying a supported method (e.g., "Dice", "NearestNeighbors", or "RandomSampling") or an instance of a local counterfactual method.
    model : object
        The machine learning model to be used for generating counterfactuals.
    train_dataset : pd.DataFrame
        The training dataset on which the counterfactual method will be fit. The dataset must contain a target column named "target".
    numeric_features_names : List[str]
        A list of feature names that are numeric.
    categorical_features_names : List[str]
        A list of feature names that are categorical.
    feat_to_vary : List[str]
        A list of features that are allowed to vary when generating counterfactuals.
    random_seed : int
        A seed for the random number generator to ensure reproducibility.
    n_most_important : int, optional
        The number of most important features to consider when generating counterfactuals (used by methods like RandomSampling), by default 15.
    n_categorical_most_frequent : int, optional
        The number of most frequent categorical values to consider when generating counterfactuals (used by methods like RandomSampling), by default 15.
    n_scalars : int, optional
        The number of scalar samples used during random sampling (used by RandomSampling), by default 1000.

    Returns:
    -------
    LocalCounterfactualMethod
        An instance of the appropriate local counterfactual method based on the input. For example, if `method` is "Dice", an instance of `DiceMethod` is returned.

    Raises:
    -------
    ValueError
        If an unsupported string is passed as the `method` argument.
    """
    if isinstance(method, str):
        if method == "Dice":
            dice = DiceMethod()
            dice.fit(
                model,
                train_dataset,
                "target",
                numeric_features_names,
                feat_to_vary,
                random_seed,
            )
            method = dice
        elif method == "NearestNeighbors":
            method = NearestNeighborMethod()
            method.fit(
                model,
                train_dataset,
                "target",
                numeric_features_names,
                feat_to_vary,
                random_seed,
            )
        elif method == "RandomSampling":
            method = RandomSampling(
                model=model,
                n_most_important=n_most_important,
                n_categorical_most_frequent=n_categorical_most_frequent,
                numerical_features=numeric_features_names,
                categorical_features=categorical_features_names,
                random_state=random_seed,
            )
            method.fit(train_dataset.drop(columns="target"), train_dataset["target"])
        else:
            raise ValueError(f"Unsupported local counterfactual method: {method}")
    else:
        method = method
    return method
