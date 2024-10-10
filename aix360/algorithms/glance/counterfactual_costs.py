from typing import Callable, List, Dict
import numpy as np
import pandas as pd


def build_dist_func_dataframe(
    X: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    n_bins: int = 10,
) -> Callable[[pd.DataFrame, pd.DataFrame], pd.Series]:
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

