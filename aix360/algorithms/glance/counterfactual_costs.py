from typing import Callable, List, Dict
import numpy as np
import pandas as pd


def build_dist_func_dataframe(
    X: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    n_bins: int = 10,
) -> Callable[[pd.DataFrame, pd.DataFrame], pd.Series]:
    """
    Builds and returns a custom distance function for computing distances between rows of two DataFrames based on specified numerical and categorical columns.

    For numerical columns, the values are first binned into intervals based on the provided number of bins (`n_bins`). 
    The distance between numerical features is computed as the sum of the absolute differences between binned values. For categorical columns, the distance is calculated as the number of mismatched categorical values.

    Parameters:
    ----------
    X : pd.DataFrame
        The reference DataFrame used to determine the bin intervals for numerical columns.
    numerical_columns : List[str]
        List of column names in `X` that contain numerical features.
    categorical_columns : List[str]
        List of column names in `X` that contain categorical features.
    n_bins : int, optional
        The number of bins to use when normalizing numerical columns, by default 10.

    Returns:
    -------
    Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
        A distance function that takes two DataFrames as input (`X1` and `X2`) and returns a Series of distances between corresponding rows in `X1` and `X2`.

    The distance function works as follows:
    - For numerical columns: the absolute differences between binned values are summed.
    - For categorical columns: the number of mismatches between values is counted.
    """
    feat_intervals = {
        col: ((max(X[col]) - min(X[col])) / n_bins) for col in numerical_columns
    }

    def bin_numericals(instances: pd.DataFrame):
        ret = instances.copy()
        for col in numerical_columns:
            ret[col] /= feat_intervals[col]
        return ret
    
    def dist_f(X1: pd.DataFrame, X2: pd.DataFrame) -> pd.Series:
        X1 = bin_numericals(X1)
        X2 = bin_numericals(X2)
        
        ret = (X1[numerical_columns] - X2[numerical_columns]).abs().sum(axis="columns")
        ret += (X1[categorical_columns] != X2[categorical_columns]).astype(int).sum(axis="columns")

        return ret
    
    return dist_f

