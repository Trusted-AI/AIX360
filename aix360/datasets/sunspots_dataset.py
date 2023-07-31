import os
import math
from io import StringIO
import requests
import pandas as pd


class SunspotDataset:
    """This dataset describes a monthly count of the number of observed sunspots for
    just over 230 years (1749-1983).

    The units are a count and there are 2,820 observations. The source of the dataset is
    credited to Andrews & Herzberg (1985).

    References:
        .. [#1] Andrews, D. F. and Herzberg, A. M., "Data: A Collection of Problems from
            Many Fields for the Student and Research Worker,"
            New York: Springer-Verlag, 1985.
        .. [#2] https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/sunspots.html
        .. [#3] https://r-data.pmagunia.com/dataset/r-dataset-package-datasets-sunspots
        .. [#4] Avishek Pal, PKS Prakash, "Practical Time Series Analysis"
            https://github.com/PacktPublishing/Practical-Time-Series-Analysis/

    """

    def __init__(
        self,
        url: str = None,
    ):
        self.data_folder = os.path.realpath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../data", "sunspots_data"
            )
        )
        self.data_file = os.path.realpath(
            os.path.join(self.data_folder, "sunspots.csv")
        )
        sunspots_url = (
            url
            if url is not None
            else "https://raw.githubusercontent.com/PacktPublishing/Practical-Time-Series-Analysis/master/Data%20Files/monthly-sunspot-number-zurich-17.csv"
        )

        if not os.path.exists(self.data_file):
            response = requests.get(sunspots_url)
            data = pd.read_csv(
                StringIO(response.text),
                skiprows=0,
                nrows=2820,
            )
            data.columns = ["time", "sunspots"]
            data["time"] = pd.to_datetime(data["time"], format="%Y-%m")

            data.to_csv(self.data_file, index=False)

    def _convert_to_date(self, val):
        month, year = math.modf(val)
        month = round(month * 12) + 1
        year = int(year)
        return "{}-{}".format(year, month)

    def load_data(self):
        """
        Prepares train and test dataframes.

        Returns:

            df (pd.DataFrame): The target time series pandas dataframe.
            schema (dict): The schema defining the fields used in the target and related
                dataframes.
        """
        # ----------------------------------------------------------------------------------
        # df - The target time series pandas dataframe.
        # ----------------------------------------------------------------------------------

        df = pd.read_csv(self.data_file, header=0)
        df.columns = ["month", "sunspots"]

        # ----------------------------------------------------------------------------------
        # schema - The schema defining the fields used in the target and
        # related dataframes.
        # ----------------------------------------------------------------------------------
        schema = {}
        schema["name"] = "sunspots"
        schema[
            "description"
        ] = "monthly count of the number of observed sunspots for just over 230 years (1749-1983)"
        schema["timestamp"] = "month"
        schema["targets"] = ["sunspots"]
        schema["frequency"] = "M"
        schema["external_regressors"] = []

        return df, schema
