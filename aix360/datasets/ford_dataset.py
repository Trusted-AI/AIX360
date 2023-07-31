import os
import requests
import pandas as pd
from zipfile import ZipFile
from io import BytesIO


class FordDataset:
    """
    This dataset describes engine noise with reading of length 500 from an automotive subsystem.
    Train and test data sets were collected in typical operating conditions, with minimal noise
    contamination.

    Train dataset consists of 1361 time series readings and test dataset consists of
    1320 readings.

    The units are a count and there are 1361 observations. The source of the dataset is
    UCR donated by A. Bagnall.

    References:
        .. [#1] Hoang Anh Dau, Eamonn Keogh, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu,
        Shaghayegh Gharghabi , Chotirat Ann Ratanamahatana, Yanping Chen, Bing Hu,
        Nurjahan Begum, Anthony Bagnall , Abdullah Mueen, Gustavo Batista, & Hexagon-ML (2019).
        The UCR Time Series Classification Archive.
        URL https://www.cs.ucr.edu/~eamonn/time_series_data_2018/.
        .. [#2] http://www.timeseriesclassification.com/description.php?Dataset=FordA

    """

    def __init__(self, url: str = None, category_a: bool = True):
        self.data_folder = os.path.realpath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../data", "ford_data"
            )
        )
        self.train_data_file = os.path.realpath(
            os.path.join(self.data_folder, "FordA_TRAIN.txt")
        )
        self.test_data_file = os.path.realpath(
            os.path.join(self.data_folder, "FordA_TEST.txt")
        )

        self.category = "A" if category_a else "B"
        ford_data_url = (
            url
            if url is not None
            else "https://timeseriesclassification.com/aeon-toolkit/Ford{}.zip".format(
                self.category
            )
        )

        self.input_length = 500
        if not os.path.exists(self.train_data_file):
            response = requests.get(ford_data_url)
            byte_content = ZipFile(BytesIO(response.content))
            byte_content.extractall(self.data_folder)

    def load_data(self):
        """
        Prepares train and test dataframes.

        Returns:

            x_train (np.ndarray): Train data in numpy format.
            x_test (np.ndarray): Test data in numpy format.
            y_train (np.ndarray): Train labels in numpy format.
            x_test (np.ndarray): Test labels in numpy format.
        """
        train = pd.read_csv(self.train_data_file, delim_whitespace=True, header=None)
        test = pd.read_csv(self.test_data_file, delim_whitespace=True, header=None)

        y_train = train.iloc[:, 0]
        x_train = train.iloc[:, 1:]
        x_test = test.iloc[:, 1:]
        y_test = test.iloc[:, 0]

        x_train = x_train.values.reshape(-1, self.input_length, 1)
        x_test = x_test.values.reshape(-1, self.input_length, 1)

        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

        return x_train, x_test, y_train, y_test
