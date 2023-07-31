import os
import requests
import numpy as np
import pandas as pd
from zipfile import ZipFile
from io import BytesIO
from tensorflow import keras


class ClimateDataset:
    """
    The dataset is from Max Planck Institute for Biogeochemistry, Keras examples.

    The dataset consists of 14 variables (temperature, humidity, wind direction,
    pressure etc.) measured every 10 minutes. The data is aggregated for an hour for analysis.
    The data is aggregated for an hour (6 samples) and selected 7 features for training.
    Further, 120 hours is used as past window and 12 hours as forecast horizon.

    References:
        .. [#1] Max Planck Institute for Biogeochemistry. https://www.bgc-jena.mpg.de/wetter/.
        .. [#2] https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_weather_forecasting.py
        .. [#3] https://www.kaggle.com/stytch16/jena-climate-2009-2016

    """

    def __init__(
        self,
        url: str = None,
    ):
        self.data_folder = os.path.realpath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../data", "climate_data"
            )
        )
        self.data_file = os.path.realpath(
            os.path.join(self.data_folder, "jena_climate_2009_2016.csv")
        )
        climate_data_url = (
            url
            if url is not None
            else "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        )

        self.input_length = 500
        # download data
        if not os.path.exists(self.data_file):
            response = requests.get(climate_data_url)
            byte_content = ZipFile(BytesIO(response.content))
            byte_content.extractall(self.data_folder)

        self.time_column = "Date Time"
        self.feature_names = [
            "Pressure",
            "Temperature",
            "Temperature in Kelvin",
            "Temperature (dew point)",
            "Relative Humidity",
            "Saturation vapor pressure",
            "Vapor pressure",
            "Vapor pressure deficit",
            "Specific humidity",
            "Water vapor concentration",
            "Airtight",
            "Wind speed",
            "Maximum wind speed",
            "Wind direction in degrees",
        ]

    def _normalize(self, data, train_split):
        data_mean = data[:train_split].mean(axis=0)
        data_std = data[:train_split].std(axis=0)
        return (data - data_mean) / data_std

    def load_data(self, return_train: bool = False, test_start: int = None):
        """
        Prepares train and test dataframes.

        Returns:

            x_train (np.ndarray): Train data in numpy format.
            x_test (np.ndarray): Test data in numpy format.
            y_train (np.ndarray): Train labels in numpy format.
            x_test (np.ndarray): Test labels in numpy format.
        """
        df = pd.read_csv(self.data_file)
        feature_columns = df.columns.tolist()
        feature_columns.remove(self.time_column)
        selected_feature_columns = [feature_columns[i] for i in [0, 1, 5, 7, 8, 10, 11]]
        selected_feature_names = [
            self.feature_names[i] for i in [0, 1, 5, 7, 8, 10, 11]
        ]
        timestamps = df[self.time_column]
        target_column = selected_feature_columns[1]
        df = df[selected_feature_columns]

        # preprocessing
        split_fraction = 0.715
        train_split = int(split_fraction * int(df.shape[0]))
        step = 6

        past = 720  # 120 hours input length
        future = 72  # 12 hours horizon
        learning_rate = 0.001
        batch_size = 1
        epochs = 10
        sequence_length = int(past / step)  # 120

        # normalization
        df[selected_feature_columns] = self._normalize(
            df[selected_feature_columns].values, train_split
        )

        # train test split
        train_data = df.iloc[0 : train_split - 1]
        test_data = df.iloc[train_split:]

        x_train = None
        y_train = None
        if return_train:
            start = past + future
            end = start + train_split

            x_train = train_data.values
            y_train = df.iloc[start:end][target_column]

            dataset_train = keras.preprocessing.timeseries_dataset_from_array(
                x_train,
                y_train,
                sequence_length=sequence_length,
                sampling_rate=step,
                batch_size=batch_size,
            )
            x_train = []
            y_train = []

            for a, b in dataset_train:
                x_train.append(a.numpy().reshape(-1, len(selected_feature_columns)))
                y_train.append(b.numpy())

            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)

        test_end = test_data.shape[0] - past - future

        label_start = train_split

        if not test_start:
            test_start = 0
            label_start = train_split
        else:
            label_start = (label_start + test_start) if test_start > 0 else test_start

        label_start = label_start + past + future

        x_test = test_data.iloc[test_start:test_end].values
        y_test = df.iloc[label_start:][target_column]

        dataset_test = keras.preprocessing.timeseries_dataset_from_array(
            x_test,
            y_test,
            sequence_length=sequence_length,
            sampling_rate=step,
            batch_size=batch_size,
        )
        x_test = []
        y_test = []

        for a, b in dataset_test:
            x_test.append(a.numpy().reshape(-1, len(selected_feature_columns)))
            y_test.append(b.numpy())

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        datasets = {
            "df": df,
            "selected_feature_columns": selected_feature_columns,
            "selected_feature_names": selected_feature_names,
            "sequence_length": sequence_length,
            "x_test": x_test,
            "y_test": y_test,
            "x_train": x_train,
            "y_train": y_train,
            "timestamps": timestamps,
        }
        return datasets
