import pandas as pd
import numpy as np
import pytest
from aix360.algorithms.glance.counterfactual_costs import   build_dist_func_dataframe
# Assume build_dist_func_dataframe is imported from the relevant module

def test_build_dist_func_dataframe():
    # Setup input DataFrame with numerical and categorical columns
    X = pd.DataFrame({
        'age': [25, 30, 22, 45],
        'salary': [50000, 60000, 55000, 53000],
        'gender': ['Male', 'Female', 'Female', 'Male']
    })

    # Specify numerical and categorical columns
    numerical_columns = ['age', 'salary']
    categorical_columns = ['gender']

    # Build the distance function
    dist_func = build_dist_func_dataframe(X, numerical_columns, categorical_columns, n_bins=5)

    # Generate test DataFrames
    X1 = pd.DataFrame({
        'age': [26, 31, 23, 46],
        'salary': [51000, 61000, 54000, 52000],
        'gender': ['Male', 'Female', 'Male', 'Female']
    })

    X2 = pd.DataFrame({
        'age': [25, 30, 22, 45],
        'salary': [50000, 60000, 55000, 53000],
        'gender': ['Female', 'Female', 'Female', 'Male']
    })

    # Calculate expected distances manually
    feat_intervals = {
        col: ((max(X[col]) - min(X[col])) / 5) for col in numerical_columns
    }


    expected_distances = np.array([
        abs(26 - 25) / feat_intervals['age'] + abs(51000 - 50000) / feat_intervals['salary'] + (X1['gender'][0] != X2['gender'][0]),  # First row: age diff + gender diff
        abs(31 - 30) / feat_intervals['age'] + abs(61000 - 60000) / feat_intervals['salary'] + (X1['gender'][1] != X2['gender'][1]),  # Second row: age diff + gender same
        abs(23 - 22) / feat_intervals['age'] + abs(54000 - 55000) / feat_intervals['salary'] + (X1['gender'][2] != X2['gender'][2]),  # Third row: age diff + gender diff
        abs(46 - 45) / feat_intervals['age'] + abs(52000 - 53000) / feat_intervals['salary'] + (X1['gender'][3] != X2['gender'][3])   # Fourth row: age diff + gender same
    ])

    # Invoke the distance function
    distances = dist_func(X1, X2)

    # Assert the distances are as expected
    pd.testing.assert_series_equal(distances, pd.Series(expected_distances), check_dtype=False)
