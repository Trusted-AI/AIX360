import pytest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List, Dict
from aix360.algorithms.glance.iterative_merges.iterative_merges import C_GLANCE,cumulative,action_fake_cost,_select_action_low_cost,_select_action_max_eff,_select_action_mean,print_results,format_glance_output,_generate_clusters, _one_hot_encode,_find_candidate_clusters,_merge_clusters
import unittest

# Sample data for testing
@pytest.fixture
def sample_data():
    data = {
        'feature1': ['A', 'B', 'A', 'B', 'C'],
        'feature2': [1, 2, 1, 2, 3],
        'feature3': [5.0, 6.0, 5.5, 6.5, 7.0]
    }
    return pd.DataFrame(data)

def test_one_hot_encode(sample_data):
    categorical_columns = ['feature1']
    encoded_df = _one_hot_encode(sample_data, categorical_columns)

    # Check if the output is a DataFrame
    assert isinstance(encoded_df, pd.DataFrame)

    # Check if the correct columns are present after encoding
    expected_columns = ['ohe__feature1_A',	'ohe__feature1_B',	'ohe__feature1_C',	'remainder__feature2',	'remainder__feature3']
    assert all(col in encoded_df.columns for col in expected_columns)

    # Check the shape of the output
    assert encoded_df.shape == (5, 5)  # 5 rows and 5 columns

    # Check the first row of the encoded DataFrame
    expected_first_row = [1.0, 0.0, 0.0, 1, 5.0]  # One-hot encoded values for first entry
    assert all(encoded_df.iloc[0].values == expected_first_row)

def test_generate_clusters(sample_data):
    categorical_features_names = ['feature1']
    num_clusters = 2

    # Create a KMeans clustering method
    clustering_method = KMeans(n_clusters=num_clusters, random_state=42)

    clusters = _generate_clusters(sample_data, num_clusters, categorical_features_names, clustering_method)

    # Check if the clusters dictionary is returned
    assert isinstance(clusters, dict)

    # Check if the number of clusters is correct
    assert len(clusters) == num_clusters

    # Check if all instances are assigned to a cluster
    all_assigned = sum(len(cluster) for cluster in clusters.values())
    assert all_assigned == sample_data.shape[0]

    # Check if the clusters are correctly assigned
    for cluster_id, cluster_df in clusters.items():
        assert isinstance(cluster_df, pd.DataFrame)
        assert all(cluster_df.index.isin(sample_data.index))

@pytest.fixture
def clusters_data():
    return {
        0: pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]}),
        1: pd.DataFrame({'feature1': [5, 6], 'feature2': [7, 8]}),
        2: pd.DataFrame({'feature1': [9], 'feature2': [10]}),  # Smallest cluster
    }

@pytest.fixture
def centroids_data():
    return {
        0: pd.DataFrame({'feature1': [1.5], 'feature2': [3.5]}),
        1: pd.DataFrame({'feature1': [5.5], 'feature2': [7.5]}),
        2: pd.DataFrame({'feature1': [9], 'feature2': [10]}),
    }

@pytest.fixture
def explanations_data():
    return {
        0: pd.DataFrame({'explanation': [0.1, 0.2]}),
        1: pd.DataFrame({'explanation': [0.3, 0.4]}),
        2: pd.DataFrame({'explanation': [0.5]}),
    }

@pytest.fixture
def distance_function():
    # A simple mock distance function
    def mock_distance(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
        return pd.Series(np.random.rand(df1.shape[0]))  # Random distances
    return mock_distance

def test_find_candidate_clusters(clusters_data, centroids_data, explanations_data, distance_function):
    heuristic_weights = (0.5, 0.5)

    # Call the function with mock data
    candidate_cluster = _find_candidate_clusters(
        clusters=clusters_data,
        cluster_centroids=centroids_data,
        explanations_centroid=explanations_data,
        heuristic_weights=heuristic_weights,
        dist_func_dataframe=distance_function
    )

    # Validate the output
    assert isinstance(candidate_cluster, tuple)
    assert len(candidate_cluster) == 2

    # The smallest cluster is expected to be cluster 2 (with only one instance)
    assert candidate_cluster[0] == 2  # Expected smallest cluster
    assert candidate_cluster[1] in [0, 1]  # The candidate cluster should be either cluster 0 or cluster 1

    # Additional checks can be added based on your specific expectations
    # For instance, we could check that the candidate cluster is one of the remaining clusters
    assert candidate_cluster[1] != candidate_cluster[0]


@pytest.fixture
def categorical_columns():
    return ["Action1", "Action2"]

def test_print_results(capsys, clusters_stats):
    total_effectiveness = 0.8
    total_cost = 250.0

    # Call the print_results function
    print_results(clusters_stats, total_effectiveness, total_cost)

    # Capture the output
    captured = capsys.readouterr()

    # Validate the output
    assert "CLUSTER 1 with size 10:" in captured.out
    assert "Effectiveness: 85.00%, Cost: 100.00" in captured.out
    assert "CLUSTER 2 with size 15:" in captured.out
    assert "Effectiveness: 75.00%, Cost: 150.00" in captured.out

@pytest.fixture
def clusters_stats():
    return {
        0: {
            "size": 10,
            "action": pd.DataFrame({"Action1": [1], "Action2": [2]}),
            "effectiveness": 0.85,
            "cost": 100.0,
        },
        1: {
            "size": 15,
            "action": pd.DataFrame({"Action1": [-1], "Action2": [0]}),
            "effectiveness": 0.75,
            "cost": 150.0,
        },
    }

@pytest.fixture
def categorical_columns():
    return ["Action1", "Action2"]

def strip_ansi_codes(text: str) -> str:
    import re

    """Remove ANSI escape sequences from a string."""
    ansi_escape = re.compile(r'\x1B\[[0-?9;]*[mK]')
    return ansi_escape.sub('', text)

def test_format_glance_output(capsys, clusters_stats, categorical_columns):
    # Convert the action DataFrames into a Series for the test
    for cluster_id in clusters_stats.keys():
        clusters_stats[cluster_id]['action'] = clusters_stats[cluster_id]['action'].iloc[0]

    # Call the format_glance_output function
    format_glance_output(clusters_stats, categorical_columns)

    # Capture the output
    captured = capsys.readouterr()

    # Strip ANSI codes from captured output for clean comparison
    output = strip_ansi_codes(captured.out)

    # Validate the output for cluster 1
    assert "Action 1" in output
    assert "Effectiveness: 85.00%" in output
    assert "Cost: 100.00" in output
    assert "Action 2" in output
    assert "Effectiveness: 75.00%" in output
    assert "Cost: 150.00" in output


class MockModel:
    def predict(self, X):
        # Simple mock predict function that flips the prediction based on a threshold
        return (X.sum(axis=1) > 1).astype(int)

def mock_distance_function(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    # Simple distance function that computes the sum of absolute differences
    return (df1 - df2).abs().sum(axis=1)

def test_select_action_mean():
    # Sample instances DataFrame
    instances = pd.DataFrame({
        'feature1': [0, 2, 1],
        'feature2': [1, 0, 1]
    })

    # Sample candidate actions DataFrame
    candidate_actions = pd.DataFrame({
        'feature1': [1, 0, 2],
        'feature2': [0, 2, 3]
    })

    # Define numerical and categorical features
    numerical_features_names = ['feature1', 'feature2']
    categorical_features_names = []  # Assuming no categorical features for this test

    # Create a mock model instance
    model = MockModel()

    # Call the _select_action_mean function
    n_flipped, recourse_cost_sum, mean_action = _select_action_mean(
        model=model,
        instances=instances,
        candidate_actions=candidate_actions,
        dist_func_dataframe=mock_distance_function,
        numerical_features_names=numerical_features_names,
        categorical_features_names=categorical_features_names,
    )
    # Expected values
    expected_n_flipped = 3  # From the mock model predictions (0 + 1 + 1 = 2)
    
    # The calculation for recourse cost sum

    expected_recourse_cost_sum = 6

    # Mean of candidate actions
    expected_mean_action = pd.Series({
        'feature1': 1.0,  
        'feature2': 1.666667   
    })
    
    mean_action = mean_action.astype('float64')
    expected_mean_action = expected_mean_action.astype('float64')
    # Assertions
    assert n_flipped == expected_n_flipped
    assert recourse_cost_sum == expected_recourse_cost_sum
    pd.testing.assert_series_equal(mean_action, expected_mean_action)

def test_select_action_max_eff():
    # Sample instances DataFrame
    instances = pd.DataFrame({
        'feature1': [0, 1, 2],
        'feature2': [1, 0, 1]
    })

    # Sample candidate actions DataFrame
    candidate_actions = pd.DataFrame({
        'feature1': [1, 0],
        'feature2': [0, 2]
    })

    # Define numerical and categorical features
    numerical_features_names = ['feature1', 'feature2']
    categorical_features_names = []  # Assuming no categorical features for this test

    # Create a mock model instance
    model = MockModel()

    # Call the _select_action_max_eff function
    max_n_flipped, recourse_cost_sum, best_action = _select_action_max_eff(
        model=model,
        instances=instances,
        candidate_actions=candidate_actions,
        dist_func_dataframe=mock_distance_function,
        numerical_features_names=numerical_features_names,
        categorical_features_names=categorical_features_names,
        num_actions=1,
    )
    
    # Expected values
    expected_max_n_flipped = 3 # Based on the mock model predictions
    expected_recourse_cost_sum = 3

    expected_best_action = pd.Series([1, 0], index=['feature1', 'feature2'], name=0)
    expected_best_action = expected_best_action.astype(np.int64)
    best_action = best_action.astype(np.int64)
    
    assert max_n_flipped == expected_max_n_flipped
    assert recourse_cost_sum == expected_recourse_cost_sum
    pd.testing.assert_series_equal(best_action, expected_best_action)

def test_select_action_low_cost():
    
    model = MockModel()
    instances = pd.DataFrame({
                    'feature1': [1, 2, 3],
                    'feature2': [0, 5, 6]
                })
                
                # Example cluster instances DataFrame
    cluster_instances = pd.DataFrame({
                    'feature1': [1, 2],
                    'feature2': [4, 5]
                })
                
    candidate_actions = pd.DataFrame({
                    'feature1': [1, 2],
                    'feature2': [3, 4],})
    
    numerical_features_names = ['feature1', 'feature2']
    categorical_features_names = []
    action_threshold = 0.5
    num_low_cost = 1
    inv_total_clusters = 1

        # Mock the dist_func_dataframe
    dist_func_dataframe = mock_distance_function


        # Call the function under test
    n_flipped, min_recourse_cost_sum, best_action = _select_action_low_cost(
        model=model,
        instances=instances,
        cluster_instances=cluster_instances,
        candidate_actions=candidate_actions,
        dist_func_dataframe=dist_func_dataframe,
        numerical_features_names=numerical_features_names,
        categorical_features_names=categorical_features_names,
        action_threshold=action_threshold,
        num_low_cost=num_low_cost,
        inv_total_clusters=inv_total_clusters,
    )

    assert n_flipped == 2
    assert min_recourse_cost_sum == 8
    pd.testing.assert_series_equal(best_action, pd.Series([1, 3], index=['feature1', 'feature2'], name=0))

class TestActionFakeCost(unittest.TestCase):

    def test_basic_functionality(self):
        # Sample action with numerical and categorical features
        action = pd.Series({
            'feature1': 10,
            'feature2': 20,
            'cat_feature1': "-",
            'cat_feature2': "value"
        })
        numerical_features_names = ['feature1', 'feature2']
        categorical_features_names = ['cat_feature1', 'cat_feature2']

        result = action_fake_cost(action, numerical_features_names, categorical_features_names)
        expected = 10 + 20 + 1  # sum of numerical + count of non "-" in categorical
        self.assertEqual(result, expected)

    def test_no_categorical_features(self):
        # Action with only numerical features
        action = pd.Series({
            'feature1': 15,
            'feature2': 25
        })
        numerical_features_names = ['feature1', 'feature2']
        categorical_features_names = []  # No categorical features

        result = action_fake_cost(action, numerical_features_names, categorical_features_names)
        expected = 15 + 25  # Just the sum of numerical features
        self.assertEqual(result, expected)

    def test_all_categorical_features(self):
        # Action with categorical features only
        action = pd.Series({
            'cat_feature1': "-",
            'cat_feature2': "-"
        })
        numerical_features_names = []  # No numerical features
        categorical_features_names = ['cat_feature1', 'cat_feature2']

        result = action_fake_cost(action, numerical_features_names, categorical_features_names)
        expected = 0  # No numerical features and both categorical are "-"
        self.assertEqual(result, expected)

    def test_mixed_features(self):
        # Action with mixed categorical features
        action = pd.Series({
            'feature1': 5,
            'feature2': 10,
            'cat_feature1': "value",
            'cat_feature2': "-"
        })
        numerical_features_names = ['feature1', 'feature2']
        categorical_features_names = ['cat_feature1', 'cat_feature2']

        result = action_fake_cost(action, numerical_features_names, categorical_features_names)
        expected = 5 + 10 + 1  # sum of numerical features + count of non "-"
        self.assertEqual(result, expected)

def mock_apply_action_pandas(instances, action, numeric_features_names, categorical_features_names, categorical_no_action_token):
    # Example: Apply an action by adding a fixed value to numeric features
    modified_instances = instances.copy()
    for feature in numeric_features_names:
        if feature in action:
            modified_instances[feature] += action[feature]
    return modified_instances

# Replace the real function with the mock one
apply_action_pandas = mock_apply_action_pandas

def test_cumulative():
    instances = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0, 5, 6]
        })
        
        # Example cluster instances DataFrame
        
    candidate_actions = [pd.Series({'feature1': 3,'feature2': 2})]
    model = MockModel()
    categorical_features_names = []
    numeric_features_names = ['feature1', 'feature2']
    categorical_no_action_token = "-"

    effectiveness, cost = cumulative(
        model,
        instances,
        candidate_actions,
        mock_distance_function,
        numeric_features_names,
        categorical_features_names,
        categorical_no_action_token
    )

    assert effectiveness == 3
    assert cost == 15

def test_iterative_merges_init():
    model = MockModel()
    im = C_GLANCE(model=model)
    
    assert im.model == model
    assert im.initial_clusters == 100
    assert im.final_clusters == 10
    assert im.num_local_counterfactuals == 5
    assert im.heuristic_weights == (0.5, 0.5)
    assert im.alternative_merges is True
    assert im.random_seed == 13
    assert im.verbose is True

def test_set_features_names():
# Sample data for testing
    sample_X = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0]
    })

    sample_y = pd.Series([0, 1, 0])  # Target variable
    model = MockModel()
    im = C_GLANCE(model=model)
    
    numerical_names, categorical_names = im._set_features_names(sample_X, None, None)
    assert numerical_names == ['feature1', 'feature2']
    assert categorical_names == []

    numerical_names, categorical_names = im._set_features_names(sample_X, ['feature1'], None)
    assert numerical_names == ['feature1']
    assert categorical_names == ['feature2']

    numerical_names, categorical_names = im._set_features_names(sample_X, None, ['feature2'])
    assert numerical_names == ['feature1']
    assert categorical_names == ['feature2']

def test_fit():
    sample_X = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0],
        'target':[0,1,1]
    })

    sample_y = pd.Series([0, 1, 0])  # Target variable
    model = MockModel()
    im = C_GLANCE(model=model)
    
    result = im.fit(sample_X.drop(columns='target'), sample_y, sample_X)
    
    assert isinstance(result, C_GLANCE)
    assert im.numerical_features_names == ['feature1', 'feature2']
    assert im.categorical_features_names == []
    assert im.X.equals(sample_X.drop(columns='target'))
    assert im.y.equals(sample_y)
    assert im.train_dataset.equals(sample_X)