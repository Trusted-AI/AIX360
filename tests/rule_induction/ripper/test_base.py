from unittest import TestCase
from sklearn.preprocessing import LabelEncoder

import pandas as pd

from aix360.algorithms.rule_induction.ripper import base


def create_test_df():
    df = pd.DataFrame({
        'int_col': [1, 2],
        'float_col': [1.2, 4.5],
        'str_col': ['foo', 'bar']
    })
    return df.astype({'int_col': 'int64', 'float_col': 'float64', 'str_col': 'str'})


class TestBase(TestCase):
    def test_init_encoder(self):
        df = create_test_df()

        actual = base.init_encoder(df)
        expected_keys = ['int_col', 'str_col']
        self.assertCountEqual(actual.keys(), expected_keys)

    def test_encode_nominal(self):
        mock_int_encoder = LabelEncoder()
        mock_int_encoder.classes_ = [1, 3]
        mock_str_encoder = LabelEncoder()
        mock_str_encoder.classes_ = ['foo', 'bar']
        label_encoders = {'int_col': mock_int_encoder, 'str_col': mock_str_encoder}
        actual_df = create_test_df()
        base.encode_nominal(label_encoders, actual_df)

        expected_df = pd.DataFrame({
            'int_col': [0, base.DEFAULT_ENCODING_VALUE],
            'float_col': [1.2, 4.5],
            'str_col': [0, 1]
        })
        self.assertTrue(actual_df.equals(expected_df))
