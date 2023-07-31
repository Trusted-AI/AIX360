import os
import math
from io import StringIO
import requests
import pandas as pd
from sklearn.model_selection import train_test_split


class DiabetesDataset:
    """This dataset consists of 10 baseline variables, age, sex, body mass index, average
    blood pressure, and six blood serum measurements were obtained for each of n = 442
    diabetes patients, as well as the response of interest, a quantitative measure of disease
    progression one year after baseline.

    References:
        .. [#1] Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani, "Least Angle Regression,"
        .. [#2] https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

    """

    def __init__(
        self,
        url: str = None,
    ):
        self.data_folder = os.path.realpath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../data", "diabetes_data"
            )
        )
        self.data_file = os.path.realpath(
            os.path.join(self.data_folder, "diabetes.csv")
        )
        diabetes_url = (
            url
            if url is not None
            else "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"
        )

        if not os.path.exists(self.data_file):
            response = requests.get(diabetes_url)
            data = pd.read_csv(StringIO(response.text), sep="\t")
            data.to_csv(self.data_file, index=False)

    def load_data(self, return_only_numerical=True, test_size=0.3, random_state=None):
        """
        Prepares train and test dataframes.

        Returns:

            x_train (np.ndarray): Train data in numpy format.
            x_test (np.ndarray): Test data in numpy format.
            y_train (np.ndarray): Train labels in numpy format.
            x_test (np.ndarray): Test labels in numpy format.
        """
        df = pd.read_csv(self.data_file)
        target_names = ["Y"]

        feature_names = df.columns.tolist()
        feature_names.remove(target_names[0])

        if return_only_numerical:
            feature_names.remove("SEX")  # categorical feature

        X = df[feature_names].values
        y = df[target_names].values.reshape(-1)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return x_train, x_test, y_train, y_test, feature_names, target_names
