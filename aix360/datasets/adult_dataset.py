import os
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def default_preprocessing(data):
    all_columns = ["Age", "Workclass", "Education", "Marital-Status",
                            "Occupation", "Relationship", "Race", "Sex", "Capital-Gain",
                            "Capital-Loss", "Hours-Per-Week", "Native-Country", "Status"]
    cate_columns = ['Workclass', 'Education', 'Marital-Status', 'Occupation',
                                    'Relationship', 'Race', 'Sex', 'Native-Country']
    numerical_columns = [c for c in all_columns if c not in cate_columns + ["Status"]]

    # remove redundant education num column (education processed in one_hot)
    data = data.drop(2, axis=1)
    data = data.drop(4, axis=1)
    # remove rows with missing values: '?,'
    data = data.replace('?,', np.nan); data = data.dropna() 
    data.columns = all_columns
    for col in data.columns[:-1]:
        #print(col)
        if col not in cate_columns:
            data[col] = data[col].apply(lambda x: float(x[:-1]))
        else:
            data[col] = data[col].apply(lambda x: x[:-1])
    # Prepocess Targets to <=50K = 0, >50K = 1
    data[data.columns[-1]] = data[data.columns[-1]].replace(['<=50K', '>50K'],
                                                            [0, 1])

    data = data.reset_index(drop=True)

    for col in numerical_columns:
        data[col] = data[col].astype(int)

    for col in data.columns:
        if col not in numerical_columns and col != data.columns[-1]:
            data[col] = data[col].astype(str)
    data = data[data['Native-Country'] != 'Holand-Netherlands']
    return data


class AdultDataset():
    """Adult Dataset.

    The Adult dataset, also known as the "Census Income" dataset, is a widely used collection of demographic information derived from the 1994 U.S. Census database 
    and is available at https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data. 
    The target variable is whether an individual earns more than $50,000 per year,
    making it a popular dataset for classification tasks in machine learning.
    """

    def __init__(self, custom_preprocessing=default_preprocessing, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','adult_data')
	
        self._filepath = os.path.join(self._dirpath, 'adult.csv')
        print("Using Adult dataset: ", self._filepath)
        
        try:
            #require access to dataframe
            #df = pd.read_csv(filepath)
            self._df = pd.read_csv(self._filepath, header = None, delim_whitespace = True)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please place the adult.csv:")
            print("file, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', 'data','adult_data'))))
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

