import pytest
import numpy as np
import pandas as pd
from statistics import multimode
from aix360.algorithms.glance.clustering import  KMeansMethod  # Adjust the import based on your module structure
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from aix360.algorithms.glance.utils.centroid import centroid_pandas,centroid_numpy  # Replace 'your_module' with the actual module name
from aix360.algorithms.glance.utils.action import apply_action_numpy,apply_action_pandas,actions_mean_pandas,apply_actions_pandas_rows,extract_actions_pandas
from aix360.algorithms.glance.utils.metadata_requests import _decide_cluster_method,_decide_local_cf_method
from aix360.algorithms.glance.local_cfs import DiceMethod, NearestNeighborMethod, RandomSampling
from xgboost import XGBClassifier

def test_centroid_pandas():
    """
    Test the centroid_pandas function for numerical and categorical columns.
    """
    data = {
        "age": [25, 30, 22, 35, 40],
        "salary": [50000, 60000, 55000, 45000, 70000],
        "gender": ["Male", "Female", "Female", "Male", "Female"],
    }
    df = pd.DataFrame(data)
    
    numerical_columns = ["age", "salary"]
    categorical_columns = ["gender"]
    
    centroid = centroid_pandas(df, numerical_columns, categorical_columns)
    
    # Expected centroid values
    expected_centroid = pd.DataFrame({
        "age": [30.4],  # Mean of [25, 30, 22, 35, 40]
        "salary": [56000.0],  # Mean of [50000, 60000, 55000, 45000, 70000]
        "gender": ["Female"],  # Mode of ['Male', 'Female', 'Female', 'Male', 'Female']
    })
    
    pd.testing.assert_frame_equal(centroid, expected_centroid)


def test_centroid_pandas_no_categorical_columns():
    """
    Test centroid_pandas when there are no categorical columns.
    """
    data = {
        "age": [25, 30, 22, 35, 40],
        "salary": [50000, 60000, 55000, 45000, 70000],
    }
    df = pd.DataFrame(data)
    
    numerical_columns = ["age", "salary"]
    categorical_columns = []
    
    centroid = centroid_pandas(df, numerical_columns, categorical_columns)
    
    # Expected centroid values (just the mean of the numerical columns)
    expected_centroid = pd.DataFrame({
        "age": [30.4],  # Mean of [25, 30, 22, 35, 40]
        "salary": [56000.0],  # Mean of [50000, 60000, 55000, 45000, 70000]
    })
    
    pd.testing.assert_frame_equal(centroid, expected_centroid)

def test_centroid_numpy():
    """
    Test the centroid_numpy function for numerical and categorical columns.
    """
    data = np.array([
        [25, 50000, 0],  # 0 = Male
        [30, 60000, 1],  # 1 = Female
        [22, 55000, 1],
        [35, 45000, 0],
        [40, 70000, 1]
    ])
    
    numerical_columns = [0, 1]  # age and salary
    categorical_columns = [2]   # gender
    
    centroid = centroid_numpy(data, numerical_columns, categorical_columns)
    
    # Expected centroid values
    expected_centroid = np.array([[30.4, 56000.0, 1]])  # Mode of gender is '1' (Female)

    np.testing.assert_array_equal(centroid, expected_centroid)


def test_centroid_numpy_no_categorical_columns():
    """
    Test centroid_numpy when there are no categorical columns.
    """
    data = np.array([
        [25, 50000],
        [30, 60000],
        [22, 55000],
        [35, 45000],
        [40, 70000]
    ])
    
    numerical_columns = [0, 1]  # age and salary
    categorical_columns = []    # No categorical columns
    
    centroid = centroid_numpy(data, numerical_columns, categorical_columns)
    
    # Expected centroid values (just the mean of the numerical columns)
    expected_centroid = np.array([[30.4, 56000.0]])

    np.testing.assert_array_equal(centroid, expected_centroid)


def test_apply_action_pandas():
    """
    Test the apply_action_pandas function for applying actions to numerical and categorical columns.
    """
    data = pd.DataFrame({
        "age": [25, 30, 22, 35],
        "salary": [50000, 60000, 55000, 45000],
        "gender": ["Male", "Female", "Female", "Male"]
    })
    
    action = pd.Series({"age": 5, "salary": 1000, "gender": "Female"})
    
    numerical_columns = ["age", "salary"]
    categorical_columns = ["gender"]
    categorical_no_action_token = "NoChange"
    
    result = apply_action_pandas(data, action, numerical_columns, categorical_columns, categorical_no_action_token)
    
    expected_result = pd.DataFrame({
        "age": [30, 35, 27, 40],  # Age incremented by 5
        "salary": [51000, 61000, 56000, 46000],  # Salary incremented by 1000
        "gender": ["Female", "Female", "Female", "Female"]  # Gender set to 'Female'
    })
    
    pd.testing.assert_frame_equal(result, expected_result)

def test_apply_action_pandas_no_change_token():
    """
    Test apply_action_pandas with a categorical no-action token.
    """
    data = pd.DataFrame({
        "age": [25, 30, 22, 35],
        "salary": [50000, 60000, 55000, 45000],
        "gender": ["Male", "Female", "Female", "Male"]
    })
    
    action = pd.Series({"age": 5, "salary": 1000, "gender": "NoChange"})
    
    numerical_columns = ["age", "salary"]
    categorical_columns = ["gender"]
    categorical_no_action_token = "NoChange"
    
    result = apply_action_pandas(data, action, numerical_columns, categorical_columns, categorical_no_action_token)
    
    expected_result = pd.DataFrame({
        "age": [30, 35, 27, 40],  # Age incremented by 5
        "salary": [51000, 61000, 56000, 46000],  # Salary incremented by 1000
        "gender": ["Male", "Female", "Female", "Male"]  # No change for gender
    })
    
    pd.testing.assert_frame_equal(result, expected_result)

def test_apply_action_numpy():
    """
    Test the apply_action_numpy function for applying actions to numerical and categorical columns.
    """
    data = np.array([
        [25, 50000, 0],  # 0 = Male
        [30, 60000, 1],  # 1 = Female
        [22, 55000, 1],
        [35, 45000, 0]
    ])
    
    action = np.array([5, 1000, 1])  # Increase age by 5, salary by 1000, gender to 'Female' (1)
    
    numerical_columns = [0, 1]
    categorical_columns = [2]
    categorical_no_action_token = 0  # '0' means no change for gender
    
    result = apply_action_numpy(data, action, numerical_columns, categorical_columns, categorical_no_action_token)
    
    expected_result = np.array([
        [30, 51000, 1],
        [35, 61000, 1],
        [27, 56000, 1],
        [40, 46000, 1]
    ])
    
    np.testing.assert_array_equal(result, expected_result)

def test_extract_actions_pandas():
    """
    Test the extract_actions_pandas function for extracting actions from differences between two dataframes.
    """
    X = pd.DataFrame({
        "age": [25, 30, 22],
        "salary": [50000, 60000, 55000],
        "gender": ["Male", "Female", "Female"]
    })
    
    cfs = pd.DataFrame({
        "age": [30, 30, 25],
        "salary": [51000, 61000, 55000],
        "gender": ["Female", "Female", "Male"]
    })
    
    numerical_features = ["age", "salary"]
    categorical_features = ["gender"]
    categorical_no_action_token = "NoChange"
    
    result = extract_actions_pandas(X, cfs, categorical_features, numerical_features, categorical_no_action_token)
    
    expected_result = pd.DataFrame({
        "age": [5, 0, 3],  # Difference in ages
        "salary": [1000, 1000, 0],  # Difference in salary
        "gender": ["Female", "NoChange", "Male"]  # Gender action, 'NoChange' for unchanged
    })
    
    pd.testing.assert_frame_equal(result, expected_result)

def test_apply_actions_pandas_rows():
    """
    Test the apply_actions_pandas_rows function for applying row-wise actions to numerical and categorical columns.
    """
    X = pd.DataFrame({
        "age": [25, 30, 22],
        "salary": [50000, 60000, 55000],
        "gender": ["Male", "Female", "Female"]
    })
    
    actions = pd.DataFrame({
        "age": [5, 0, 3],
        "salary": [1000, 1000, 0],
        "gender": ["Female", "NoChange", "Male"]
    })
    
    numerical_columns = ["age", "salary"]
    categorical_columns = ["gender"]
    categorical_no_action_token = "NoChange"
    
    result = apply_actions_pandas_rows(X, actions, numerical_columns, categorical_columns, categorical_no_action_token)
    
    expected_result = pd.DataFrame({
        "age": [30, 30, 25],  # Age updated by actions
        "salary": [51000, 61000, 55000],  # Salary updated by actions
        "gender": ["Female", "Female", "Male"]  # Gender updated where applicable
    })
    
    pd.testing.assert_frame_equal(result, expected_result)

def test_actions_mean_pandas():
    """
    Test the actions_mean_pandas function for calculating the mean action for numerical and categorical columns.
    """
    actions = pd.DataFrame({
        "age": [5, 0, 3],
        "salary": [1000, 1000, 0],
        "gender": ["Female", "NoChange", "Male"]
    })
    
    numerical_features = ["age", "salary"]
    categorical_features = ["gender"]
    categorical_no_action_token = "NoChange"
    
    result = actions_mean_pandas(actions, numerical_features, categorical_features, categorical_no_action_token)
    
    expected_result = pd.Series({
        "age": 8 / 3,  # Mean of [5, 0, 3]
        "salary": 2000 / 3,  # Mean of [1000, 1000, 0]
        "gender": "Female"  # Most frequent value
    })
    
    pd.testing.assert_series_equal(result, expected_result)

def test_decide_cluster_method_kmeans():
    """Test that 'KMeans' method returns KMeansMethod instance."""
    n_clusters = 3
    random_seed = 42
    result = _decide_cluster_method("KMeans", n_clusters, random_seed)
    assert isinstance(result, KMeansMethod)
    assert result.num_clusters == n_clusters
    assert result.random_seed == random_seed

def test_decide_cluster_method_invalid():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported clustering method: unsupported_method"):
        _decide_cluster_method("unsupported_method", n_clusters=3, random_seed=42)

def test_decide_cluster_method_instance():
    """Test that passing an instance returns the same instance."""
    kmeans_instance = KMeansMethod(num_clusters=3, random_seed=42)
    result = _decide_cluster_method(kmeans_instance, n_clusters=None, random_seed=None)
    assert result is kmeans_instance

def test_decide_local_cf_method_dice():
    """Test that 'Dice' method returns an instance of DiceMethod."""
    model = XGBClassifier()  # Create or mock a model as required
    train_dataset = X = pd.DataFrame({
        "age": [25, 30, 22],
        "salary": [50000, 60000, 55000],
        "gender": ["Male", "Female", "Female"],
        'target': [1,0,1]
    })  # Create or mock a DataFrame as required
    numeric_features_names = ['age', 'salary']
    categorical_features_names = ['gender']
    feat_to_vary = ['age', 'salary']
    random_seed = 42

    result = _decide_local_cf_method("Dice", model, train_dataset, numeric_features_names, categorical_features_names, feat_to_vary, random_seed)
    assert isinstance(result, DiceMethod)

def test_decide_local_cf_method_nearest_neighbors():
    """Test that 'NearestNeighbors' method returns an instance of NearestNeighborMethod."""
    model = XGBClassifier()  # Create or mock a model as required
    train_dataset = X = pd.DataFrame({
        "age": [25, 30, 22,45,60,20],
        "salary": [50000, 60000, 55000, 53000,75000,30000],
        "gender": ["Male", "Female", "Female","Male", "Female",'Female'],
        'target': [1,0,1,0,1,1]
    }) 
    numeric_features_names = ['age', 'salary']
    categorical_features_names = ['gender'] 
    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                                categorical_features_names,
                            )
                        ],
                        remainder="passthrough",
                    )
    model_ = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("normalize", StandardScaler()),
            ("classifier", model),
        ]
    )
    model_.fit(
        train_dataset.drop(columns='target'),
        train_dataset['target'],
    )
    feat_to_vary = ['age', 'salary']
    random_seed = 42

    result = _decide_local_cf_method("NearestNeighbors", model_, train_dataset, numeric_features_names, categorical_features_names, feat_to_vary, random_seed)
    assert isinstance(result, NearestNeighborMethod)

def test_decide_local_cf_method_random_sampling():
    """Test that 'RandomSampling' method returns an instance of RandomSampling."""
    model = XGBClassifier()  # Create or mock a model as required
    train_dataset = X = pd.DataFrame({
        "age": [25, 30, 22,45,60,20],
        "salary": [50000, 60000, 55000, 53000,75000,30000],
        "gender": ["Male", "Female", "Female","Male", "Female",'Female'],
        'target': [1,0,1,0,1,1]
    }) 
    numeric_features_names = ['age', 'salary']
    categorical_features_names = ['gender'] 
    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                                categorical_features_names,
                            )
                        ],
                        remainder="passthrough",
                    )
    model_ = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("normalize", StandardScaler()),
            ("classifier", model),
        ]
    )
    model_.fit(
        train_dataset.drop(columns='target'),
        train_dataset['target'],
    )
    feat_to_vary = ['age', 'salary']
    random_seed = 42

    result = _decide_local_cf_method("RandomSampling", model_, train_dataset, numeric_features_names, categorical_features_names, feat_to_vary, random_seed)
    assert isinstance(result, RandomSampling)

def test_decide_local_cf_method_invalid():
    """Test that invalid method raises ValueError."""
    model = XGBClassifier()  # Create or mock a model as required
    train_dataset = X = pd.DataFrame({
        "age": [25, 30, 22],
        "salary": [50000, 60000, 55000],
        "gender": ["Male", "Female", "Female"],
        'target': [1,0,1]
    })  # Create or mock a DataFrame as required
    numeric_features_names = ['age', 'salary']
    categorical_features_names = ['gender']
    feat_to_vary = ['age', 'salary']
    random_seed = 42

    with pytest.raises(ValueError, match="Unsupported local counterfactual method: unsupported_method"):
        _decide_local_cf_method("unsupported_method", model, train_dataset, numeric_features_names, categorical_features_names, feat_to_vary, random_seed)

def test_decide_local_cf_method_instance():
    """Test that passing an instance returns the same instance."""
    dice_instance = DiceMethod()
    model = XGBClassifier()  # Create or mock a model as required
    train_dataset = X = pd.DataFrame({
        "age": [25, 30, 22],
        "salary": [50000, 60000, 55000],
        "gender": ["Male", "Female", "Female"],
        'target': [1,0,1]
    })  # Create or mock a DataFrame as required
    numeric_features_names = ['age', 'salary']
    categorical_features_names = ['gender']
    feat_to_vary = ['age', 'salary']
    random_seed = 42

    result = _decide_local_cf_method(dice_instance, model, train_dataset, numeric_features_names, categorical_features_names, feat_to_vary, random_seed)
    assert result is dice_instance