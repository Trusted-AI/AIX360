import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from aix360.algorithms.glance.local_cfs.nearest_neighbor import NearestNeighborMethod  
from aix360.algorithms.glance.local_cfs.random_sampling import RandomSampling  
from aix360.algorithms.glance.local_cfs.dice_method import DiceMethod  


# Create a sample dataset for testing
def create_sample_data():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['outcome'] = y
    return df

# Test for NearestNeighborMethod
def test_nearest_neighbor_method():
    # Create sample data
    data = create_sample_data()
    
    # Initialize model and methods
    model = LogisticRegression()
    model.fit(data.drop(columns='outcome'), data['outcome'])  # Fit model to data
    nn_method = NearestNeighborMethod()
    
    # Fit the NearestNeighborMethod
    nn_method.fit(model, data, outcome_name='outcome', continuous_features=[f'feature_{i}' for i in range(5)], feat_to_vary=['feature_0'])

    # Test explaining instances
    instances = data.sample(5).drop(columns='outcome')  # Randomly select 5 instances
    counterfactuals = nn_method.explain_instances(instances, num_counterfactuals=3)

    # Validate output
    assert counterfactuals.shape[0] == 15  # 5 instances * 3 counterfactuals
    assert set(counterfactuals.columns) == set(data.columns[:-1])  # Check if columns match original features



def create_sample_data_random(num_samples=100):
    np.random.seed(42)  # For reproducibility
    data = pd.DataFrame({
        'feature_0': np.random.rand(num_samples),
        'feature_1': np.random.rand(num_samples),
        'feature_2': np.random.rand(num_samples),
        'feature_3': np.random.rand(num_samples),
        'feature_4': np.random.rand(num_samples),
        'outcome': np.random.choice([0, 1], num_samples)
    })
    return data

@pytest.fixture
def setup_data():
    """Fixture to create sample data and fit a model."""
    data = create_sample_data_random()
    model = LogisticRegression()
    model.fit(data.drop(columns='outcome'), data['outcome'])
    
    return model, data

def test_random_sampling_method(setup_data):
    model, data = setup_data
    
    # Initialize RandomSampling method
    nn_method = RandomSampling(model, n_most_important=3, n_categorical_most_frequent=2,
                               numerical_features=[f'feature_{i}' for i in range(5)], 
                               categorical_features=[])

    # Fit the method
    nn_method.fit(data.drop(columns='outcome'), data['outcome'])

    # Test explaining instances
    instances = data.sample(5).drop(columns='outcome')  # Randomly select 5 instances
    counterfactuals = nn_method.explain_instances(instances, num_counterfactuals=3)

    # Validate output
    assert counterfactuals.shape[0] == 15  # 5 instances * 3 counterfactuals
    assert set(counterfactuals.columns) == set(data.columns[:-1])  # Check if columns match original features

def test_invalid_explain_input_shape(setup_data):
    model, data = setup_data
    
    nn_method = RandomSampling(model, n_most_important=3, n_categorical_most_frequent=2,
                               numerical_features=[f'feature_{i}' for i in range(5)], 
                               categorical_features=[])
    nn_method.fit(data.drop(columns='outcome'), data['outcome'])
    
    # Check that ValueError is raised for empty DataFrame
    with pytest.raises(ValueError, match="Input must be a single row DataFrame."):
        nn_method.explain(pd.DataFrame(), num_counterfactuals=3)

def test_explain_instances_with_all_one_class(setup_data):
    model, data = setup_data
    
    # Create a subset with only one class
    all_zeros = data[data['outcome'] == 0].sample(5).drop(columns='outcome')
    nn_method = RandomSampling(model, n_most_important=3, n_categorical_most_frequent=2,
                               numerical_features=[f'feature_{i}' for i in range(5)], 
                               categorical_features=[])
    nn_method.fit(data.drop(columns='outcome'), data['outcome'])
    
    # Explain instances where all instances belong to one class
    counterfactuals = nn_method.explain_instances(all_zeros, num_counterfactuals=3)
    
    # Assert that the output is handled properly, could be empty or valid based on implementation
    assert isinstance(counterfactuals, pd.DataFrame)

def test_explain_instances_with_insufficient_valid_counterfactuals(setup_data):
    model, data = setup_data
    
    # Testing behavior when the method should return fewer counterfactuals than requested
    nn_method = RandomSampling(model, n_most_important=3, n_categorical_most_frequent=2,
                               numerical_features=[f'feature_{i}' for i in range(5)], 
                               categorical_features=[])
    nn_method.fit(data.drop(columns='outcome'), data['outcome'])
    
    few_instances = data.sample(1).drop(columns='outcome')  # Only one instance
    counterfactuals = nn_method.explain_instances(few_instances, num_counterfactuals=10)
    
    # The output should not exceed the number requested
    assert counterfactuals.shape[0] <= 10
# def test_random_sampling_method():
#     data = create_sample_data()
    
#     # Initialize model and methods
#     model = LogisticRegression()
#     model.fit(data.drop(columns='outcome'), data['outcome'])  # Fit model to data
#     nn_method = RandomSampling(model, 15, 20, numerical_features=[f'feature_{i}' for i in range(5)], categorical_features=[])
#     nn_method.fit(data.drop(columns='outcome'), data['outcome'])

#     instances = data.sample(5).drop(columns='outcome')  # Randomly select 5 instances
#     counterfactuals = nn_method.explain_instances(instances, num_counterfactuals=3)
#     # Validate output
#     assert counterfactuals.shape[0] == 15  # 5 instances * 3 counterfactuals
#     assert set(counterfactuals.columns) == set(data.columns[:-1])  # Check if columns match original features

def test_dice_method():
    data = create_sample_data()
    
    # Initialize model and methods
    model = LogisticRegression()
    model.fit(data.drop(columns='outcome'), data['outcome'])  # Fit model to data
    dice_method = DiceMethod()
    dice_method.fit(model,data,'outcome',[f'feature_{i}' for i in range(5)],[f'feature_{i}' for i in range(2)])
    instances = data.sample(5).drop(columns='outcome')
    counterfactuals = dice_method.explain_instances(instances, num_counterfactuals=3)
    # Validate output
    assert counterfactuals.shape[0] == 15  # 5 instances * 3 counterfactuals
    assert set(counterfactuals.columns) == set(data.columns[:-1])  # Check if columns match original features
