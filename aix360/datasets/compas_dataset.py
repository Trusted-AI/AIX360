import os
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def default_preprocessing(data):
    data = data.dropna(subset=["days_b_screening_arrest"])
    data = data.rename(columns={data.columns[-1]: "status"})
    data = data.to_dict("list")
    for k in data.keys():
        data[k] = np.array(data[k])

    dates_in = data["c_jail_in"]
    dates_out = data["c_jail_out"]
    # this measures time in Jail
    time_served = []
    for i in range(len(dates_in)):
        di = datetime.datetime.strptime(dates_in[i], "%Y-%m-%d %H:%M:%S")
        do = datetime.datetime.strptime(dates_out[i], "%Y-%m-%d %H:%M:%S")
        time_served.append((do - di).days)
    time_served = np.array(time_served)
    time_served[time_served < 0] = 0
    data["time_served"] = time_served

    """ Filtering the data """
    # These filters are as taken by propublica
    # (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days
    # from when the person was arrested, we assume that because of data quality
    # reasons, that we do not have the right offense.
    idx = np.logical_and(
        data["days_b_screening_arrest"] <= 30, data["days_b_screening_arrest"] >= -30
    )

    # We coded the recidivist flag -- is_recid -- to be -1
    # if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of
    # 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O")
    # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows
    # representing people who had either recidivated in two years, or had at least two
    # years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]
    data = pd.DataFrame(data)
    cols = [
            "Sex",
            "Age_Cat",
            "Race",
            "C_Charge_Degree",
            "Priors_Count",
            "Time_Served",
            "Status",
    ]
    data = data[[col.lower() for col in cols]]
    data.columns = cols
    return data


class COMPASDataset():
    """COMPAS Dataset.

    The COMPAS dataset (Correctional Offender Management Profiling for Alternative Sanctions) Angwin et al. (2016) 
    is available at https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv. 
    Detailed description and information on the dataset can be found at https://www.propublica.org/
    article/how-we-analyzed-the-compas-recidivism-algorithm. It categorizes recidivism risk 
    based on several factors, including race.



    """

    def __init__(self, custom_preprocessing=default_preprocessing, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','compas_data')
	
        self._filepath = os.path.join(self._dirpath, 'compas.csv')
        print("Using Compas dataset: ", self._filepath)
        
        try:
            #require access to dataframe
            #df = pd.read_csv(filepath)
            self._df = pd.read_csv(self._filepath)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please place the compas.csv:")
            print("file, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', 'data','compas_data'))))
            import sys
            sys.exit(1)

        if custom_preprocessing:
            #require access to dataframe
            #self._data = custom_preprocessing(df)
            self._data = custom_preprocessing(self._df.copy())

    # return a copy of the dataframe with Riskperformance as last column
    def dataframe(self):
        # First pop and then add 'Riskperformance' column
        dfcopy = self._data.copy()
        return(dfcopy)

    def data(self):
        return self._data

