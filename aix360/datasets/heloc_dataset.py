import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical

def nan_preprocessing(df):
    # Separate and encode target variable as 0,1
    y = df.pop('RiskPerformance')
    enc = LabelEncoder()
    y = enc.fit_transform(y)
    # Remove rows where all data is missing
    y = y[(df > -9).any(axis=1)]
    df = df.loc[(df > -9).any(axis=1)]
    # Replace remaining special values with NaN
    df = df[df >= 0]
    df['RiskPerformance'] = y
    return df

def default_preprocessing(df):
    # Details and preprocessing for FICO dataset

    # minimize dependence on ordering of columns in heloc data
    # x_cols, y_col = df.columns[0:-1], df.columns[-1]
    x_cols = list(df.columns.values)
    x_cols.remove('RiskPerformance')
    y_col = list(['RiskPerformance'])

    # Preprocessing the HELOC dataset
    # Remove all the rows containing -9 in the ExternalRiskEstimate column
    # df = df[df.ExternalRiskEstimate != -9]
    # add columns for -7 and -8 in the dataset
    for col in x_cols:
        df[col][df[col].isin([-7, -8, -9])] = 0
    # Get the column names for the covariates and the dependent variable
    df = df[(df[x_cols].T != 0).any()]

    # minimize dependence on ordering of columns in heloc data
    # x = df.values[:, 0:-1]
    x = df[x_cols].values

    # encode target variable ('bad', 'good')
    cat_values = df[y_col].values
    enc = LabelEncoder()
    enc.fit(cat_values)
    num_values = enc.transform(cat_values)
    y = np.array(num_values)
    
    return np.hstack((x, y.reshape(y.shape[0], 1)))


class HELOCDataset():
    """HELOC Dataset.

    The FICO HELOC dataset [#]_ contains anonymized information about home equity line of credit (HELOC) applications 
    made by real homeowners. A HELOC is a line of credit typically offered by a US bank as a percentage of home
    equity (the difference between the current market value of a home and the outstanding balance of all liens,
    e.g. mortgages). The customers in this dataset have requested a credit line in the range of USD 5,000 - 150,000.

    The target variable in this dataset is a binary variable called RiskPerformance. The value “Bad” indicates that an
    applicant was 90 days past due or worse at least once over a period of 24 months from when the credit account
    was opened. The value “Good” indicates that they have made their payments without ever being more than 90 days
    overdue.

    This dataset can be used to train a machine learning model to predict whether the homeowner qualifies for a line
    of credit or not. The HELOC dataset and more information about it, including instructions to download are available in
    the reference below.

    References:
        .. [#] `Explainable Machine Learning Challenge - FICO Community.
           <https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=2>`_

    """

    def __init__(self, custom_preprocessing=default_preprocessing, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','heloc_data')
	
        self._filepath = os.path.join(self._dirpath, 'heloc_dataset.csv')
        print("Using Heloc dataset: ", self._filepath)
        
        try:
            #require access to dataframe
            #df = pd.read_csv(filepath)
            self._df = pd.read_csv(self._filepath)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please place the heloc_dataset.csv:")
            print("file, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', 'data','heloc_data'))))
            import sys
            sys.exit(1)

        if custom_preprocessing:
            #require access to dataframe
            #self._data = custom_preprocessing(df)
            self._data = custom_preprocessing(self._df.copy())

    # return a copy of the dataframe with Riskperformance as last column
    def dataframe(self):
        # First pop and then add 'Riskperformance' column
        dfcopy = self._df.copy()
        col = dfcopy.pop('RiskPerformance')
        dfcopy['RiskPerformance'] = col
        return(dfcopy)

    def data(self):
        return self._data

    def split(self, random_state=0):
        (data_train, data_test) = train_test_split(self._data, stratify=self._data[:,-1], random_state=random_state)

        x_train = data_train[:,0:-1]
        x_test = data_test[:, 0:-1]
        y_train = data_train[:, -1]
        y_test = data_test[:, -1]

	#y_train_b = to_categorical(y_train)
	#y_test_b = to_categorical(y_test)

        # convert y_train and y_test into onehot encoded form
        enc = OneHotEncoder().fit(self._data[:,-1].reshape(-1,1))
        y_train_b = enc.transform(y_train.reshape(-1,1)).toarray()
        y_test_b = enc.transform(y_test.reshape(-1,1)).toarray()

        return (self._data, x_train, x_test, y_train_b, y_test_b)
