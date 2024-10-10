from typing import List, Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd


def apply_action_pandas(
    X: pd.DataFrame,
    action: pd.Series,
    numerical_columns: List[str],
    categorical_columns: List[str],
    categorical_no_action_token: Any,
    numerical_no_action_token: Optional[Any] = None,
) -> pd.DataFrame:
    """Apply `action` to all rows of `X`. For numerical columns, add the
    respective component from `action`. For categorical columns, set the
    component of all rows to the value of `action`, unless it is equal to
    the `categorical_no_action_token`, in which case do nothing for this
    column.

    Args:
        X (pd.DataFrame): matrix of observations
        action (pd.Series): for each column / feature, the action to be applied
        numerical_columns (List[str]): numerical column names
        categorical_columns (List[str]): categorical column names
        categorical_no_action_token (Any): special value signifying no-action (i.e. equivalent to 0 for numerical columns)

    Returns:
        pd.DataFrame: new observations resulting from the action application.
    """
    assert (X.columns == action.index).all()
    if numerical_no_action_token is None:
        numerical_no_action_token = categorical_no_action_token

    ret = X.copy(deep=True)
    for col in numerical_columns:
        if action[col] != numerical_no_action_token:
            ret[col] = X[col] + action[col]
    for col in categorical_columns:
        if action[col] != categorical_no_action_token:
            ret[col] = action[col]
    ret = ret.astype(X.dtypes)

    return ret


def apply_action_numpy(
    X: npt.NDArray[np.number],
    action: npt.NDArray[np.number],
    numerical_columns: List[int],
    categorical_columns: List[int],
    categorical_no_action_token: np.number,
) -> npt.NDArray[np.number]:
    """Apply `action` to all rows of `X`. For numerical columns, add the
    respective component from `action`. For categorical columns, set the
    component of all rows to the value of `action`, unless it is equal to
    the `categorical_no_action_token`, in which case do nothing for this
    column.

    Note: input array should have a numeric dtype. Thus, categorical columns
    should be encoded by numbers (e.g. Ordinal Encoding).

    Args:
        X (npt.NDArray[np.number]): matrix of observations
        action (npt.NDArray[np.number]): for each column / feature, the action to be applied
        numerical_columns (List[int]): numerical column indices
        categorical_columns (List[int]): categorical column indices
        categorical_no_action_token (np.number): special value signifying no-action (i.e. equivalent to 0 for numerical columns)

    Returns:
        npt.NDArray[np.number]: new observations resulting from the action application.
    """
    assert len(X.shape) == 2
    assert len(action.shape) == 1
    assert (
        X.shape[1] == action.shape[0]
    ), "action should have length equal to the number of columns"

    ret = X.copy()
    ret[:, numerical_columns] += action[numerical_columns]
    categorical_columns_masked = np.intersect1d(
        np.where(action != categorical_no_action_token)[0], categorical_columns
    )
    ret[:, categorical_columns_masked] = action[categorical_columns_masked]

    return ret


def extract_actions_pandas(
    X: pd.DataFrame,
    cfs: pd.DataFrame,
    categorical_features: List[str],
    numerical_features: List[str],
    categorical_no_action_token: Any,
):
    actions = X.copy(deep=True)

    for col in categorical_features:
        are_equal_indicator = X[col] == cfs[col]
        actions.loc[are_equal_indicator, col] = categorical_no_action_token
        actions.loc[~are_equal_indicator, col] = cfs.loc[~are_equal_indicator, col]
    for col in numerical_features:
        actions[col] = cfs[col] - X[col]
    return actions

def apply_actions_pandas_rows(
    X: pd.DataFrame,
    actions: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    categorical_no_action_token: object,
) -> pd.DataFrame:
    ret = X.copy(deep=True)
    for col in numerical_columns:
        ret[col] = X[col] + actions[col]
    for col in categorical_columns:
        no_action_indicator = actions[col] == categorical_no_action_token
        ret.loc[~ no_action_indicator, col] = actions.loc[~ no_action_indicator, col].values
        ret.loc[no_action_indicator, col] = X.loc[no_action_indicator, col].values

    return ret

def actions_mean_pandas(
    actions: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    categorical_no_action_token: Any,
) -> pd.Series:
    ret = pd.Series(index=actions.columns, dtype="object")
    ret[numerical_features] = actions[numerical_features].mean()
    for col in categorical_features:
        if (actions[col] == categorical_no_action_token).all():
            ret[col] = categorical_no_action_token
        else:
            value_cnts = actions[col].value_counts()
            most_freq = (
                value_cnts.index[0]
                if value_cnts.index[0] != categorical_features
                else value_cnts.index[1]
            )
            ret[col] = most_freq

    return ret
