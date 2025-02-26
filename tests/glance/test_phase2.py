import pytest
import pandas as pd
import numpy as np
from aix360.algorithms.glance.iterative_merges.phase2 import generate_cluster_centroid_explanations
from aix360.algorithms.glance.utils.action import extract_actions_pandas
from aix360.algorithms.glance.utils.centroid import centroid_pandas
from aix360.algorithms.glance.base import LocalCounterfactualMethod
from unittest.mock import MagicMock

@pytest.fixture
def setup_cluster_data():
    # Create synthetic cluster centroids
    cluster_centroids = {
        0: pd.DataFrame(np.random.rand(1, 5), columns=[f'feature_{i}' for i in range(5)]),
        1: pd.DataFrame(np.random.rand(1, 5), columns=[f'feature_{i}' for i in range(5)])
    }
    numerical_features_names = [f'feature_{i}' for i in range(3)]
    categorical_features_names = [f'feature_{i}' for i in range(3, 5)]
    return cluster_centroids, numerical_features_names, categorical_features_names


@pytest.fixture
def setup_mock_cf_generator():
    # Mock the LocalCounterfactualMethod and its explain_instances method
    cf_generator = MagicMock(spec=LocalCounterfactualMethod)
    
    # Simulate counterfactual generation returning valid data
    def mock_explain_instances(instances, num_counterfactuals):
        return pd.DataFrame(
            np.random.rand(num_counterfactuals, instances.shape[1]),
            columns=instances.columns
        )
    
    cf_generator.explain_instances.side_effect = mock_explain_instances
    return cf_generator


def test_generate_cluster_centroid_explanations_basic(setup_cluster_data, setup_mock_cf_generator):
    cluster_centroids, numerical_features_names, categorical_features_names = setup_cluster_data
    cf_generator = setup_mock_cf_generator

    num_local_counterfactuals = 3

    cluster_explanations, cluster_expl_actions, explanations_centroid = generate_cluster_centroid_explanations(
        cluster_centroids=cluster_centroids,
        cf_generator=cf_generator,
        num_local_counterfactuals=num_local_counterfactuals,
        numerical_features_names=numerical_features_names,
        categorical_features_names=categorical_features_names
    )

    # Test the shape and type of returned cluster_explanations
    assert isinstance(cluster_explanations, dict)
    assert len(cluster_explanations) == len(cluster_centroids)
    for cluster_id, cf in cluster_explanations.items():
        assert isinstance(cf, pd.DataFrame)
        assert cf.shape[0] == num_local_counterfactuals

    # Test the cluster_expl_actions are returned and not empty
    assert isinstance(cluster_expl_actions, dict)
    assert len(cluster_expl_actions) == len(cluster_centroids)

    # Test the centroid calculations
    assert isinstance(explanations_centroid, dict)
    assert len(explanations_centroid) == len(cluster_centroids)
    for cluster_id, centroid in explanations_centroid.items():
        # Fix: Expecting DataFrame instead of Series
        assert isinstance(centroid, pd.DataFrame)  # Update to DataFrame
        assert centroid.shape[1] == len(numerical_features_names) + len(categorical_features_names)


def test_generate_cluster_centroid_explanations_empty_counterfactuals(setup_cluster_data):
    cluster_centroids, numerical_features_names, categorical_features_names = setup_cluster_data

    # Mock the LocalCounterfactualMethod to return empty DataFrames for counterfactuals
    cf_generator = MagicMock(spec=LocalCounterfactualMethod)
    cf_generator.explain_instances.return_value = pd.DataFrame()

    num_local_counterfactuals = 3

    with pytest.raises(ValueError, match="No counterfactuals found for any of the centroids."):
        generate_cluster_centroid_explanations(
            cluster_centroids=cluster_centroids,
            cf_generator=cf_generator,
            num_local_counterfactuals=num_local_counterfactuals,
            numerical_features_names=numerical_features_names,
            categorical_features_names=categorical_features_names
        )


def test_generate_cluster_centroid_explanations_incorrect_num_counterfactuals(setup_cluster_data, setup_mock_cf_generator):
    cluster_centroids, numerical_features_names, categorical_features_names = setup_cluster_data
    cf_generator = setup_mock_cf_generator

    # Mock the explain_instances method to return an incorrect number of counterfactuals
    def mock_explain_instances(instances, num_counterfactuals):
        return pd.DataFrame(np.random.rand(2, instances.shape[1]), columns=instances.columns)  # Only return 2 cfs
    
    cf_generator.explain_instances.side_effect = mock_explain_instances

    num_local_counterfactuals = 3

    cluster_explanations, cluster_expl_actions, explanations_centroid = generate_cluster_centroid_explanations(
        cluster_centroids=cluster_centroids,
        cf_generator=cf_generator,
        num_local_counterfactuals=num_local_counterfactuals,
        numerical_features_names=numerical_features_names,
        categorical_features_names=categorical_features_names
    )

    # Test the shape and type of returned cluster_explanations
    assert isinstance(cluster_explanations, dict)
    assert len(cluster_explanations) == len(cluster_centroids)
    for cluster_id, cf in cluster_explanations.items():
        assert isinstance(cf, pd.DataFrame)
        assert cf.shape[0] != num_local_counterfactuals