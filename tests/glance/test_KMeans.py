import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from aix360.algorithms.glance.clustering import KMeansMethod  # Replace 'your_module' with the actual module name
from sklearn.exceptions import NotFittedError

def test_kmeans_initialization():
    """
    Test that KMeansMethod initializes correctly with the given number of clusters and random seed.
    """
    num_clusters = 3
    random_seed = 42
    kmeans_method = KMeansMethod(num_clusters=num_clusters, random_seed=random_seed)
    
    assert kmeans_method.num_clusters == num_clusters
    assert kmeans_method.random_seed == random_seed
    assert isinstance(kmeans_method.model, KMeans)  # Ensure model is an instance of KMeans


def test_kmeans_fit():
    """
    Test that the KMeansMethod can fit data properly.
    """
    # Create synthetic data
    num_clusters = 3
    data, _ = make_blobs(n_samples=100, centers=num_clusters, random_state=42)
    
    kmeans_method = KMeansMethod(num_clusters=num_clusters, random_seed=42)
    kmeans_method.fit(data)
    
    assert kmeans_method.model.n_clusters == num_clusters
    assert hasattr(kmeans_method.model, 'cluster_centers_')  # Ensure model has been fit


def test_kmeans_predict():
    """
    Test that KMeansMethod can predict cluster assignments for new instances.
    """
    # Create synthetic data
    num_clusters = 3
    data, labels = make_blobs(n_samples=100, centers=num_clusters, random_state=42)
    
    kmeans_method = KMeansMethod(num_clusters=num_clusters, random_seed=42)
    kmeans_method.fit(data)  # Fit the model first
    
    # Test prediction on new instances
    new_instances = np.array([[0, 0], [5, 5], [-5, -5]])
    predictions = kmeans_method.predict(new_instances)
    print(predictions.dtype)
    
    assert len(predictions) == len(new_instances)  # Ensure prediction length matches input length
    assert all(isinstance(pred, np.integer) for pred in predictions)  # Ensure predictions are valid cluster labels


def test_kmeans_predict_without_fit():
    """
    Test that predict raises an exception if the model hasn't been fit.
    """
    kmeans_method = KMeansMethod(num_clusters=3, random_seed=42)
    
    with pytest.raises(NotFittedError):  # Raises error when model is not fitted
        kmeans_method.predict([[0, 0]])

