""" A pandas DataFrame for time series data.
"""
from typing import List, Union
import numpy as np
import pandas as pd


def tsFrame(
    df=Union[pd.DataFrame, np.ndarray],
    timestamp_column: Union[str, int] = 0,
    columns: Union[List[str], List[int]] = None,
    freq: str = "infer",
    dt: Union[float, int] = None,  # is this required?
) -> pd.DataFrame:
    """Convert a pandas DataFrame to a time series data tsFrame.

    A time series associates values with points in time. We represent a time series as
    a tsFrame. A tsFrame is a pandas DataFrame that is indexed by Timestamp objects,
    that is, DatetimeIndex.

    Args:
        df (Union[pd.DataFrame, np.ndarray]): The input pandas DataFrame or 2D numpy array.
        timestamp_column (Union[str, int]): The name (or position) of the timestamp column.
            By default it uses the first column name. Defaults to 0.
        columns (Union[List[str], List[int]]): The names (or positions) of the columns of
            the time series values. If None, retains all the available columns except the
            timestamp column which is used as index. Defaults to None.
        freq (str): One of pandas date offset strings or corresponding objects. The string
            ‘infer’ can be passed in order to set the frequency of the index as the inferred
            frequency upon creation. https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html
            Defaults to 'infer'.
        dt (Union[float, int]): Datetime index. If None, value is set to 1. Defaults to None.

    Examples:

        Input pandas DataFrame::

                    date   sales
            0    2016-12-26    21.0
            1    2017-01-02   179.0
            2    2017-01-09   190.0
            3    2017-01-16   364.0
            4    2017-01-23   667.0
            ..          ...     ...
            153  2019-12-02  2560.0
            154  2019-12-09  3603.0
            155  2019-12-16  4719.0
            156  2019-12-23  7055.0
            157  2019-12-30  1668.0

        [158 rows x 2 columns]

        Ouput pandas DataFrame (tsFrame)::

                        sales
            date
            2016-12-26    21.0
            2017-01-02   179.0
            2017-01-09   190.0
            2017-01-16   364.0
            2017-01-23   667.0
            ...            ...
            2019-12-02  2560.0
            2019-12-09  3603.0
            2019-12-16  4719.0
            2019-12-23  7055.0
            2019-12-30  1668.0

            [158 rows x 1 columns]

    """
    if isinstance(df, np.ndarray):
        assert (
            len(df.shape) == 2
        ), f"Error: expects two dimensional sequence (n_obs x n_vars)!"
        n_obs, n_vars = df.shape
        var_names = [f"X_{i+1}" for i in range(n_vars)]
        ts = pd.DataFrame(df, columns=var_names)
        if dt is None:
            dt = 1
        if isinstance(dt, list) and len(dt) == n_obs:
            ts["time"] = dt
        else:
            ts["time"] = np.arange(n_obs) * dt
        ts.index = pd.DatetimeIndex(ts["time"], freq=freq)
        ts.drop(columns="time", inplace=True)
    else:
        ts = df.copy()
        if type(timestamp_column) == int:
            timestamp_column = df.columns[timestamp_column]

        ts.index = pd.DatetimeIndex(ts[timestamp_column], freq=freq)
        ts.drop(columns=timestamp_column, inplace=True)

        if columns is not None:
            columns = [df.columns[col] if type(col) == int else col for col in columns]
            ts = ts[columns]
    return ts


def to_np_array(
    ts: Union["tsFrame", np.ndarray],
    target_vars: Union[List[str], List[int]] = None,
):
    if target_vars is None:
        if isinstance(ts, np.ndarray):
            if len(ts.shape) == 1:
                ts = ts.reshape(-1, 1)
        target_vars = np.arange(ts.shape[1]).tolist()

    if len(ts.shape) != 2:
        raise RuntimeError(
            f"Error: expects data with 2 dimensions received {len(ts.shape)}"
        )

    if isinstance(ts, pd.DataFrame):
        target_vars = [
            ts.columns[tgt] if isinstance(tgt, int) else tgt for tgt in target_vars
        ]
        assert (
            all([tgt in ts for tgt in target_vars]) and len(target_vars) > 0
        ), "Error: improper variable specification!"
        ts_array = ts[target_vars].values
    else:
        ts_array = ts[:, target_vars]
    return ts_array
