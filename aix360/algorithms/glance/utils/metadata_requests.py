from ..base import ClusteringMethod, LocalCounterfactualMethod
from ..clustering import KMeansMethod
from ..local_cfs import DiceMethod, NearestNeighborMethod, RandomSampling


def _decide_cluster_method(method, n_clusters, random_seed) -> ClusteringMethod:
    if isinstance(method, str):
        if method == "KMeans":
            method = KMeansMethod(num_clusters=n_clusters, random_seed=random_seed)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    else:
        method = method
    return method


def _decide_local_cf_method(
    method, model, train_dataset, numeric_features_names, categorical_features_names, feat_to_vary, random_seed, n_most_important: int = 15, n_categorical_most_frequent: int = 15, n_scalars: int = 1000,
) -> LocalCounterfactualMethod:
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
