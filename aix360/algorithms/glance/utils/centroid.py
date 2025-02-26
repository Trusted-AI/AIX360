from typing import List
import pandas as pd
import numpy as np
import numpy.typing as npt
from statistics import multimode
from IPython.display import display


def centroid_pandas(
    X: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
) -> pd.DataFrame:
    """Calculates the centroid of the rows of a pandas DataFrame. Specifically,
    for the `numerical_columns` columns, the centroid has value the mean of all
    rows, while for the `categorical_columns` columns, the centroid has value
    the mode of all rows.

    Args:
        X (pd.DataFrame): matrix of observations
        numerical_columns (List[str]): numerical column names
        categorical_columns (List[str]): categorical column names

    Returns:
        pd.DataFrame: DataFrame whose single row is the centroid
    """
    centroid = pd.DataFrame(columns=X.columns).astype(X.dtypes)

    centroid.loc[0, numerical_columns] = X[numerical_columns].mean(axis="index")
    if categorical_columns != []:
        centroid.loc[0, categorical_columns] = X[categorical_columns].apply(
            lambda col: multimode(col)[0]
        )
        # centroid.loc[0, categorical_columns] = X[categorical_columns].mode().iloc[0]

    return centroid


def centroid_numpy(
    X: npt.NDArray[np.number],
    numerical_columns: List[int],
    categorical_columns: List[int],
) -> npt.NDArray[np.number]:
    """Calculates the centroid of the rows of a 2d numy array. Specifically,
    for the `numerical_columns` columns, the centroid has value the mean of all
    rows, while for the `categorical_columns` columns, the centroid has value
    the mode of all rows.

    Args:
        X (npt.NDArray[np.number]): matrix of observations
        numerical_columns (List[int]): numerical column indices
        categorical_columns (List[int]): categorical column indices

    Returns:
        npt.NDArray[np.number]: 2d numpy array whose single row is the centroid
    """
    assert len(X.shape) == 2
    centroid = np.zeros((1, X.shape[1]))

    centroid[:, numerical_columns] = X[:, numerical_columns].mean(axis=0)

    def most_frequent(x):
        unique_values, counts = np.unique(x, return_counts=True)
        most_common = unique_values[np.argmax(counts)]
        return most_common

    centroid[:, categorical_columns] = [
        most_frequent(X[:, i]) for i in categorical_columns
    ]

    return centroid
