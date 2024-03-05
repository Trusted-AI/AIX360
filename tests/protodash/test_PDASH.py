import unittest

from sklearn.preprocessing import OneHotEncoder
import numpy as np

from aix360.algorithms.protodash import ProtodashExplainer
from aix360.algorithms.protodash import get_Gaussian_Data
from aix360.datasets.cdc_dataset import CDCDataset


class TestProtoDashExplainer(unittest.TestCase):

    def test_ProtoDashExplainer(self):

        # load NHANES CDC Questionnaire data
        nhanes = CDCDataset()

        # load income questionnaire dataset
        df = nhanes.get_csv_file('INQ_H.csv')
        print(df.head(5))

        data = df.to_numpy()
        np.set_printoptions(suppress=True)
        print(data[:5])

        # One-hot encode the data
        original = data

        # replace nan's with 0's
        original[np.isnan(original)] = 0

        # delete 1st column (contains sequence numbers)
        original = original[:, 1:]

        # one hot encoding of all columns/features
        onehot_encoder = OneHotEncoder(sparse_output=False)
        onehot_encoded = onehot_encoder.fit_transform(original)

        print(onehot_encoded.shape, original.shape)

        # Obtain an explanation for data
        Y = onehot_encoded
        X = Y
        explainer = ProtodashExplainer()
        (W, S, _) = explainer.explain(X, Y, m=5)

        dfs = df.iloc[S].copy()
        dfs["Weight"] = W
        print(dfs)

        # Gaussian (simulated) data example
        # generate normalized gaussian data X, Y with 100 features and 300 & 4000 observations respectively

        (X, Y) = get_Gaussian_Data(100, 300, 4000)
        print(X.shape, Y.shape)
        (W, S, _) = explainer.explain(X, Y, m=5, kernelType='Gaussian', sigma=2)

        print(S, W)

if __name__ == '__main__':
    unittest.main()
