import sys

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from aix360.algorithms.rule_induction.ripper.binding import EQ, LE, GE

DEFAULT_ENCODING_VALUE = -sys.maxsize - 1


def _encoding_for_parallel(
        data: DataFrame,
        column_name_index_map
):
    # each condition can be indexed from those four list
    name_list = []
    op_list = []
    num_val_list = []
    nom_val_list = []

    for name in data.columns.values:
        if np.issubdtype(data.dtypes[name], np.signedinteger):
            for val in np.unique(data.loc[:, name]):
                name_list.append(column_name_index_map[name]), op_list.append(EQ), num_val_list.append(
                    -1), nom_val_list.append(val)

        elif np.issubdtype(data.dtypes[name], np.floating):
            for val in np.unique(data.loc[:, name]):
                name_list.append(column_name_index_map[name]), op_list.append(GE), num_val_list.append(
                    val), nom_val_list.append(-1)
                name_list.append(column_name_index_map[name]), op_list.append(LE), num_val_list.append(
                    val), nom_val_list.append(-1)
        else:
            raise TypeError('unsupported type {type}'.format(type=data.dtypes[name]))

    return [np.array(name_list, dtype=np.int64),
            np.array(op_list, dtype=np.uint8),
            np.array(num_val_list, dtype=np.float64),
            np.array(nom_val_list, dtype=np.int64)]


def init_encoder(
        data,
):
    label_encoders = dict()

    for name in data.columns.values:
        if np.issubdtype(data.dtypes[name], np.object_) or np.issubdtype(data.dtypes[name], np.signedinteger):
            label_encoders[name] = LabelEncoder()
            label_encoders[name].fit(data[name])

    return label_encoders


def encode_nominal(
        label_encoders,
        data,
):
    for col in label_encoders.keys():
        label_map = {val: label for label, val in enumerate(label_encoders[col].classes_)}
        data[col] = data[col].map(label_map)
        # fillna and convert to int
        data[col] = data[col].fillna(DEFAULT_ENCODING_VALUE).astype(np.int64)


def _split_instances(
        pos: np.ndarray,
        neg: np.ndarray,
        ratio,
        random_state
):
    pos_train, pos_prune = train_test_split(pos, random_state=random_state, train_size=ratio)
    neg_train, neg_prune = train_test_split(neg, random_state=random_state, train_size=ratio)

    return pos_train, pos_prune, neg_train, neg_prune
