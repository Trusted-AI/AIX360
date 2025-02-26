import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from aix360.algorithms.glance.counterfactual_tree.counterfactual_tree import T_GLANCE  # Adjust the import as needed

# Sample dataset for testing
def create_sample_data():
    # Example with 100 samples and 5 features
    np.random.seed(0)  # For reproducibility
    data = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    data['target'] = np.random.randint(0, 2, size=100)  # Binary outcome
    return data

@pytest.fixture
def sample_data():
    return create_sample_data()

@pytest.fixture
def fitted_model(sample_data):
    model = LogisticRegression()
    model.fit(sample_data.drop(columns='target'), sample_data['target'])
    return model

def test_counterfactual_tree_initialization(fitted_model):
    tree = T_GLANCE(model=fitted_model)
    assert tree.model == fitted_model
    assert tree.split_features is None
    assert tree.partition_counterfactuals is None

def test_counterfactual_tree_fit(fitted_model):
    data = create_sample_data()
    tree = T_GLANCE(model=fitted_model)
    
    # Test default split_features (None)
    tree.fit(data.drop(columns='target'), data['target'], data)
    assert len(tree.split_features) == 2  # Assuming model has informative features

    # Test with specific number of split features
    tree = T_GLANCE(model=fitted_model, split_features=3)
    tree.fit(data.drop(columns='target'), data['target'], data)
    assert len(tree.split_features) == 3

    # Test with a numeric feature list
    numeric_features = [f'feature_{i}' for i in range(5)]
    tree.fit(data.drop(columns='target'), data['target'], data ,numeric_features_names=numeric_features)
    assert tree.numerical_features_names == numeric_features

    # Test with categorical features
    categorical_features = [f'feature_0']
    tree.fit(data.drop(columns='target'), data['target'], data, categorical_features_names=categorical_features)
    assert tree.categorical_features_names == categorical_features

def test_partition_group(fitted_model):
    data = create_sample_data()
    tree = T_GLANCE(model=fitted_model)
    tree.fit(data.drop(columns='target'), data['target'],data)
    
    # Simulate instances for partitioning
    instances = data.sample(20).drop(columns='target')
    node = tree.partition_group(instances)
    
    assert node is not None
    assert hasattr(node, 'split_feature')
    assert hasattr(node, 'children')


