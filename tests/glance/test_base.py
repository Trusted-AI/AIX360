import pytest
import pandas as pd
import numpy as np
from aix360.algorithms.glance.base import ClusteringMethod, LocalCounterfactualMethod, GlobalCounterfactualMethod


# Concrete implementation for ClusteringMethod
class SimpleKMeans(ClusteringMethod):
    def __init__(self, **kwargs):
        super().__init__()  # Call parent __init__ (optional but explicit)
        self.kwargs = kwargs  # Store any passed kwargs
    
    def fit(self, data: pd.DataFrame):
        self.data = data
        self.labels = np.random.randint(0, 2, size=len(data))  # Randomly assign clusters (2 clusters)
    
    def predict(self, instances: pd.DataFrame) -> np.ndarray:
        return self.labels  # Return the same random labels


# Concrete implementation for LocalCounterfactualMethod
class SimpleCounterfactual(LocalCounterfactualMethod):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs  # Store any passed kwargs

    def fit(self, **kwargs):
        # Store additional fit kwargs
        self.fit_kwargs = kwargs
    
    def explain_instances(self, instances: pd.DataFrame, num_counterfactuals: int) -> pd.DataFrame:
        return instances.copy()  # Dummy return for testing


# Concrete implementation for GlobalCounterfactualMethod
class SimpleGlobalCounterfactual(GlobalCounterfactualMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Call parent __init__ and pass kwargs
        self.kwargs = kwargs  # Store any passed kwargs

    def fit(self, X, y, **kwargs):
        self.X = X
        self.y = y
        self.fit_kwargs = kwargs  # Store any additional kwargs
    
    def explain_group(self, instances: pd.DataFrame = None) -> pd.DataFrame:
        if instances is None:
            return self.X  # Return training data if no instances provided
        return instances.copy()


# Sample data for testing
sample_data = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0],
    'feature2': [4.0, 5.0, 6.0],
})

sample_target = pd.Series([0, 1, 0])  # Example target variable


# Test for SimpleKMeans
def test_simple_kmeans():
    kmeans = SimpleKMeans(param1="test")
    kmeans.fit(sample_data)
    
    # Check if labels are assigned correctly
    assert isinstance(kmeans, ClusteringMethod)
    assert len(kmeans.labels) == len(sample_data)
    assert set(kmeans.labels).issubset({0, 1})  # Expect labels to be either 0 or 1

    predictions = kmeans.predict(sample_data)
    assert len(predictions) == len(sample_data)  # Predictions should match the number of samples

    # Test that kwargs are passed correctly
    assert kmeans.kwargs['param1'] == "test"


# Test for SimpleCounterfactual
def test_simple_counterfactual():
    counterfactual = SimpleCounterfactual(param1="test")
    counterfactual.fit(param2="fit_param")
    
    cf = counterfactual.explain_instances(sample_data, num_counterfactuals=2)
    
    # Check if the counterfactuals are returned correctly
    assert isinstance(counterfactual, LocalCounterfactualMethod)
    assert cf.equals(sample_data)  # For this stub implementation, they should be the same

    # Test that kwargs are passed correctly
    assert counterfactual.kwargs['param1'] == "test"
    assert counterfactual.fit_kwargs['param2'] == "fit_param"


# Test for SimpleGlobalCounterfactual
def test_simple_global_counterfactual():
    global_cf = SimpleGlobalCounterfactual(param1="test")
    global_cf.fit(sample_data, sample_target, param2="fit_param")
    
    # Test case when instances are passed
    cf_group = global_cf.explain_group(sample_data)
    assert cf_group.equals(sample_data)  # For this stub implementation, they should be the same
    assert isinstance(global_cf, GlobalCounterfactualMethod)
    # Test case when no instances are passed
    cf_group_default = global_cf.explain_group()
    assert cf_group_default.equals(sample_data)  # Should return the training data
    
    # Test that kwargs are passed correctly
    assert global_cf.kwargs['param1'] == "test"
    assert global_cf.fit_kwargs['param2'] == "fit_param"


# Test abstract class instantiation errors
def test_abstract_classes_instantiation():
    with pytest.raises(TypeError):
        ClusteringMethod()  # Should raise TypeError because it is abstract

    with pytest.raises(TypeError):
        LocalCounterfactualMethod()  # Should raise TypeError because it is abstract

    with pytest.raises(TypeError):
        GlobalCounterfactualMethod()  # Should raise TypeError because it is abstract


# # Test edge case for empty data
# def test_empty_data():
#     empty_data = pd.DataFrame()
    
#     kmeans = SimpleKMeans()
#     kmeans.fit(empty_data)
#     assert len(kmeans.predict(empty_data)) == 0  # Ensure it handles empty data correctly

#     counterfactual = SimpleCounterfactual()
#     cf = counterfactual.explain_instances(empty_data, num_counterfactuals=2)
#     assert cf.empty  # Check that the result is an empty DataFrame

#     global_cf = SimpleGlobalCounterfactual()
#     global_cf.fit(empty_data, pd.Series())
#     cf_group = global_cf.explain_group(empty_data)
#     assert cf_group.empty  # Check that the result is an empty DataFrame

# def test_data_with_missing_values():
#     data_with_nan = pd.DataFrame({
#         'feature1': [1.0, np.nan, 3.0],
#         'feature2': [4.0, 5.0, np.nan],
#     })

#     # Test for ClusteringMethod
#     kmeans = SimpleKMeans()
#     kmeans.fit(data_with_nan)
#     predictions = kmeans.predict(data_with_nan)
#     assert len(predictions) == len(data_with_nan)

#     # Test for LocalCounterfactualMethod
#     counterfactual = SimpleCounterfactual()
#     cf = counterfactual.explain_instances(data_with_nan, num_counterfactuals=2)
#     assert cf.equals(data_with_nan)

#     # Test for GlobalCounterfactualMethod
#     global_cf = SimpleGlobalCounterfactual()
#     global_cf.fit(data_with_nan, pd.Series([0, 1, 0]))
#     cf_group = global_cf.explain_group(data_with_nan)
#     assert cf_group.equals(data_with_nan)

# def test_data_with_mixed_types():
#     mixed_data = pd.DataFrame({
#         'feature1': [1, 2, 3],  # integers
#         'feature2': [1.5, 2.5, 3.5],  # floats
#         'feature3': ['a', 'b', 'c']  # strings
#     })

#     # Test for ClusteringMethod
#     kmeans = SimpleKMeans()
#     kmeans.fit(mixed_data)
#     predictions = kmeans.predict(mixed_data)
#     assert len(predictions) == len(mixed_data)

#     # Test for LocalCounterfactualMethod
#     counterfactual = SimpleCounterfactual()
#     cf = counterfactual.explain_instances(mixed_data, num_counterfactuals=2)
#     assert cf.equals(mixed_data)

#     # Test for GlobalCounterfactualMethod
#     global_cf = SimpleGlobalCounterfactual()
#     global_cf.fit(mixed_data, pd.Series([0, 1, 0]))
#     cf_group = global_cf.explain_group(mixed_data)
#     assert cf_group.equals(mixed_data)

# def test_counterfactual_num_values():
#     # Test with num_counterfactuals=0
#     counterfactual = SimpleCounterfactual()
#     cf_zero = counterfactual.explain_instances(sample_data, num_counterfactuals=0)
#     assert cf_zero.equals(sample_data)  # Expected to return the same data

#     # Test with negative num_counterfactuals
#     cf_negative = counterfactual.explain_instances(sample_data, num_counterfactuals=-1)
#     assert cf_negative.equals(sample_data)  # Should handle gracefully (same data)

# def test_single_instance():
#     single_instance = pd.DataFrame({'feature1': [1.0], 'feature2': [4.0]})

#     # Test for ClusteringMethod
#     kmeans = SimpleKMeans()
#     kmeans.fit(single_instance)
#     predictions = kmeans.predict(single_instance)
#     assert len(predictions) == 1

#     # Test for LocalCounterfactualMethod
#     counterfactual = SimpleCounterfactual()
#     cf = counterfactual.explain_instances(single_instance, num_counterfactuals=1)
#     assert cf.equals(single_instance)

#     # Test for GlobalCounterfactualMethod
#     global_cf = SimpleGlobalCounterfactual()
#     global_cf.fit(single_instance, pd.Series([0]))
#     cf_group = global_cf.explain_group(single_instance)
#     assert cf_group.equals(single_instance)

# def test_fit_with_kwargs():
#     # ClusteringMethod with extra kwargs
#     kmeans = SimpleKMeans(param1="test")
#     kmeans.fit(sample_data)
#     assert kmeans.kwargs['param1'] == "test"

#     # LocalCounterfactualMethod with extra kwargs
#     counterfactual = SimpleCounterfactual(param1="test")
#     counterfactual.fit(param2="fit_param", extra_param="extra")
#     assert counterfactual.fit_kwargs['extra_param'] == "extra"

#     # GlobalCounterfactualMethod with extra kwargs
#     global_cf = SimpleGlobalCounterfactual(param1="test")
#     global_cf.fit(sample_data, sample_target, param2="fit_param", another_param="another")
#     assert global_cf.fit_kwargs['another_param'] == "another"

# def test_global_counterfactual_with_none():
#     global_cf = SimpleGlobalCounterfactual()
#     global_cf.fit(sample_data, sample_target)
    
#     # Test case with instances as None
#     cf_group_default = global_cf.explain_group(None)
#     assert cf_group_default.equals(sample_data)  # Should return training data

# def test_predict_with_empty_data():
#     empty_data = pd.DataFrame()

#     kmeans = SimpleKMeans()
#     kmeans.fit(empty_data)
#     predictions = kmeans.predict(empty_data)
#     assert len(predictions) == 0  # Ensure predict returns no labels for empty data
# def test_clustering_method_abstract_instantiation():
#     with pytest.raises(TypeError):
#         ClusteringMethod()

# def test_local_counterfactual_method_abstract_instantiation():
#     with pytest.raises(TypeError):
#         LocalCounterfactualMethod()

# def test_global_counterfactual_method_abstract_instantiation():
#     with pytest.raises(TypeError):
#         GlobalCounterfactualMethod()
