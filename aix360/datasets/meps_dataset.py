import os

import pandas as pd


def default_preprocessing(df):
    df = df[df['RACEV2X'].isin(['1','2'])]
    df = df[df['HISPANX'] == 2]

    racedict = {1.0: 0.0,2.0: 1.0}
    df = df.assign(RACEV2X = df['RACEV2X'].replace(to_replace = racedict))
    df = df.rename(columns = {'RACEV2X' : 'RACE3'})

    genderdict = {1.0: 0.0,2.0: 1.0}
    df = df.assign(SEX = df['SEX'].replace(to_replace = genderdict))
    df = df.rename(columns = {'SEX' : 'GENDER'})
    df = df.rename(columns = {'PERWT15F' : 'PERSONWT'})
    df = df.rename(columns = {'REGION31' : 'REGION'})
    df = df.rename(columns = {'TTLP15X' : 'INCOME_M'})
    df = df.rename(columns = {'TOTEXP15' : 'HEALTHEXP'})

    # #df = df.drop('HISPANX',axis=1)  .column dropping can be taken care of by simply not including in features_to_keep
    # df = df[df['PANEL'] == 19]
    # #df = df.drop('PANEL',axis=1)
    # lessE = df['TOTEXP15'] <= 5000.0
    # df.loc[lessE,'TOTEXP15'] = 0.0
    # moreE = df['TOTEXP15'] > 5000.0
    # df.loc[moreE,'TOTEXP15'] = 1.0

    df = df[df['REGION'] >= 0] # remove values -1
    df = df[df['AGE31X'] >= 0] # remove values -1

    df = df[df['MARRY31X'] >= 0] # remove values -1, -7, -8, -9
    df = df[df['INCOME_M'] >= 0]

    df = df[(df[['EDRECODE','FTSTU31X','ACTDTY31','HONRDC31','RTHLTH31','MNHLTH31','HIBPDX','CHDDX','ANGIDX',
                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON31','CHOLDX','CANCERDX','DIABDX',
                 'JTPAIN31','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT31','WLKLIM31',
                 'ACTLIM31','SOCLIM31','COGLIM31','DFHEAR42','DFSEE42','ADSMOK42',
                 'PHQ242','EMPST31','POVCAT15','INSCOV15']] >= -1).all(1)]  #for all other categorical features, remove values < -1
    df = df[['PANEL', 'REGION','AGE31X','GENDER','RACE3','MARRY31X',     # dropped 'EDUCYR', 'HIDEG' as data distributions are weird across panels,
             'EDRECODE','FTSTU31X','ACTDTY31','HONRDC31','RTHLTH31','MNHLTH31','HIBPDX','CHDDX','ANGIDX',
             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON31','CHOLDX','CANCERDX','DIABDX',
             'JTPAIN31','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT31','WLKLIM31',
             'ACTLIM31','SOCLIM31','COGLIM31','DFHEAR42','DFSEE42','ADSMOK42','PCS42',
             'MCS42','K6SUM42','PHQ242','EMPST31','POVCAT15','INSCOV15','INCOME_M','HEALTHEXP','PERSONWT']]

    df['HEALTHEXP'].mean(), df['INCOME_M'].mean()
    return df

class MEPSDataset():
    """
    The Medical Expenditure Panel Survey (MEPS) [#]_ data consists of large scale surveys of families and individuals, 
    medical providers, and employers, and collects data on health services used, costs & frequency of services,
    demographics, health status and conditions, etc., of the respondents.

    This specific dataset contains MEPS survey data for calendar year 2015 obtained in rounds 3, 4, and 5 of Panel 19,
    and rounds 1, 2, and 3 of Panel 20.
    See :file:`aix360/data/meps_data/README.md` for more details on the dataset and instructions on downloading/processing the data.

    References:
        .. [#] `Medical Expenditure Panel Survey data <https://meps.ahrq.gov/mepsweb/>`_

    """

    def __init__(self, custom_preprocessing=default_preprocessing, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','meps_data')

        self._filepath = os.path.join(self._dirpath, 'h181.csv')
        try:
            df = pd.read_csv(self._filepath, sep=',', na_values=[])
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please place the heloc_dataset.csv:")
            print("file, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', 'data','meps_data'))))
            import sys
            sys.exit(1)

        if custom_preprocessing:
            self._data = custom_preprocessing(df)

    def data(self):
        return self._data
