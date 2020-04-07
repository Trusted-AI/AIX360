import warnings
warnings.filterwarnings('ignore')

import pickle
from time import time

import numpy as np
import pandas as pd
import shap
from aix360.algorithms.rbm import BooleanRuleCG, BRCGExplainer, FeatureBinarizer, FeatureBinarizerFromTrees, \
    GLRMExplainer, LogisticRuleRegression
from aix360.datasets import HELOCDataset, MEPSDataset
from aix360.datasets.heloc_dataset import nan_preprocessing
from pandas import DataFrame
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


__all__ = ['fbt_vs_fb', 'fbt_vs_fb_cancer', 'fbt_vs_fb_crime', 'fbt_vs_fb_fico', 'fbt_vs_fb_meps', 'format_results',
           'get_corr_columns', 'print_metrics']


def fbt_vs_fb(X, y, categorical=[], iterations=30, treeNum=1, treeDepth=4, numThresh=9, filename=None):

    def fit_transform(transformer, args_train, args_test):
        X_train_fb, X_train_std_fb = transformer.fit_transform(*args_train)
        X_test_fb, X_test_std_fb = transformer.transform(*args_test)
        return X_train_fb, X_train_std_fb, X_test_fb, X_test_std_fb


    def fit_score(explainer, y_test, args_train, args_test):
        t = time()
        explainer.fit(*args_train)
        t = time() - t
        y_pred = explainer.predict(*args_test)
        if isinstance(explainer, BRCGExplainer):
            z: DataFrame = explainer._model.z.loc[:, explainer._model.w > 0.5]
            rules = z.shape[1]
            clauses = z.any(axis=1).sum()
        else:
            z: DataFrame = explainer._model.z
            rules = z.any(axis=1).sum()
            clauses = z.any(axis=1).sum() + 1  # +1 for intercept
        return (t, accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred),
                f1_score(y_test, y_pred), rules, clauses, str(explainer.explain()))


    columns = ['time', 'accuracy', 'precision', 'recall', 'f1', 'rules', 'clauses']
    index = pd.MultiIndex.from_product((['brcg', 'logrr'], ['fb', 'fbt'], range(iterations)))
    d = DataFrame(np.zeros((iterations * 4, len(columns))), dtype=float, index=index, columns=columns)
    d['explanation'] = ''

    fb = FeatureBinarizer(colCateg=categorical, negations=True, returnOrd=True, numThresh=numThresh)
    fbt = FeatureBinarizerFromTrees(colCateg=categorical, treeNum=treeNum, treeDepth=treeDepth, returnOrd=True)

    brcg = BRCGExplainer(BooleanRuleCG(silent=True))
    logrr = GLRMExplainer(LogisticRuleRegression(lambda0=0.005, lambda1=0.001, useOrd=True, maxSolverIter=1000))

    for i in range(iterations):

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=i)

        # FeatureBinarizer
        X_train_fb, X_train_std_fb, X_test_fb, X_test_std_fb = fit_transform(fb, (X_train, ), (X_test, ))
        d.loc[('brcg', 'fb', i)] = fit_score(brcg, y_test, (X_train_fb, y_train), (X_test_fb, ))
        d.loc[('logrr', 'fb', i)] = fit_score(logrr, y_test, (X_train_fb, y_train, X_train_std_fb), (X_test_fb, X_test_std_fb))

        # FeatureBinarizerFromTrees
        X_train_fb, X_train_std_fb, X_test_fb, X_test_std_fb = fit_transform(fbt, (X_train, y_train), (X_test, ))
        d.loc[('brcg', 'fbt', i)] = fit_score(brcg, y_test, (X_train_fb, y_train), (X_test_fb, ))
        d.loc[('logrr', 'fbt', i)] = fit_score(logrr, y_test, (X_train_fb, y_train, X_train_std_fb), (X_test_fb, X_test_std_fb))

    if filename is not None:
        with open(filename, 'wb') as fl:
            pickle.dump(d, fl)

    return d


def fbt_vs_fb_cancer(iterations=30, treeNum=1, treeDepth=4, numThresh=9, filename=None):
    bc = load_breast_cancer()
    X = pd.DataFrame(bc.data, columns=bc.feature_names)
    y = bc.target
    X.drop(columns=get_corr_columns(X), inplace=True)
    return fbt_vs_fb(X, y, iterations=iterations, treeNum=treeNum, treeDepth=treeDepth, numThresh=numThresh,
                     filename=filename)


def fbt_vs_fb_crime(iterations=30, treeNum=1, treeDepth=4, numThresh=9, filename=None):
    X, y = shap.datasets.communitiesandcrime()
    y = (y >= np.percentile(y, 75)).astype(np.int)
    X.drop(columns=get_corr_columns(X), inplace=True)
    return fbt_vs_fb(X, y, iterations=iterations, treeNum=treeNum, treeDepth=treeDepth, numThresh=numThresh,
                     filename=filename)


def fbt_vs_fb_fico(iterations=30, treeNum=1, treeDepth=4, numThresh=9, filename=None):
    # Load FICO HELOC data with special values converted to np.nan
    data: DataFrame = HELOCDataset(custom_preprocessing=nan_preprocessing).data()

    # FeatureBinarizerFromTrees requires the user to decide how to handle NaN/None.
    s = data.isnull().sum()
    for col in s[s > 0].index:
        data[f'{col}-NaN'] = data[col].isnull().to_numpy()
    data.fillna(-9999, inplace=True)

    # Drop highly correlated columns
    data.drop(columns=get_corr_columns(data), inplace=True)
    y = data.pop('RiskPerformance')
    return fbt_vs_fb(X, y, iterations=iterations, treeNum=treeNum, treeDepth=treeDepth, numThresh=numThresh,
                     filename=filename)


def fbt_vs_fb_meps(iterations=30, treeNum=1, treeDepth=4, numThresh=9, filename=None):
    X: pd.DataFrame = MEPSDataset().data()
    X.reset_index(drop=True, inplace=True)
    # Drop panel number (not meant to be predictive) and sample weights
    X.drop(columns=['PANEL', 'PERSONWT'], inplace=True)
    y = X.pop('HEALTHEXP')
    categorical = ['REGION', 'MARRY31X', 'EDRECODE', 'FTSTU31X', 'ACTDTY31', 'HONRDC31', 'RTHLTH31', 'MNHLTH31',
                   'HIBPDX', 'CHDDX', 'ANGIDX', 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON31', 'CHOLDX', 'CANCERDX',
                   'DIABDX', 'JTPAIN31', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT31', 'WLKLIM31', 'ACTLIM31',
                   'SOCLIM31', 'COGLIM31', 'DFHEAR42', 'DFSEE42', 'ADSMOK42', 'PHQ242', 'EMPST31', 'POVCAT15',
                   'INSCOV15']
    X.drop(columns=get_corr_columns(X), inplace=True)
    y = (y > y.mean()).astype(int)
    return fbt_vs_fb(X, y, iterations=iterations, treeNum=treeNum, treeDepth=treeDepth, numThresh=numThresh,
                     filename=filename)


def format_results(df: DataFrame, decimals=3):

    def fmt(v):
        a = [f'({vv})' for vv in v]
        return np.array(a, dtype=np.object)

    df = df.iloc[:, 0:7]
    m = df.groupby(level=(0, 1)).mean().round(decimals).astype(str)
    s: pd.DataFrame = df.groupby(level=(0, 1)).std().round(decimals).astype(str)
    s = s.transform(fmt, axis=1)
    df = pd.concat((m, s), axis=1)
    df = df.iloc[:, [0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13]]
    df.columns = pd.MultiIndex.from_product((m.columns, ['mean', 'std']))
    return df


def get_corr_columns(df: pd.DataFrame, threshold: float = 0.98) -> list:
    c = df.corr().abs()
    c: pd.DataFrame = c * np.tri(c.shape[0], c.shape[1], -1)
    c = c.transpose()
    return [col for col in c.columns if any(c[col] > threshold)]


def print_metrics(y_truth, y_pred):
    print(f'Accuracy = {accuracy_score(y_truth, y_pred):0.5f}')
    print(f'Precision = {precision_score(y_truth, y_pred):0.5f}')
    print(f'Recall = {recall_score(y_truth, y_pred):0.5f}')
    print(f'F1 = {f1_score(y_truth, y_pred):0.5f}')
    print(f'F1 Weighted = {f1_score(y_truth, y_pred, average="weighted"):0.5f}')
    print()
