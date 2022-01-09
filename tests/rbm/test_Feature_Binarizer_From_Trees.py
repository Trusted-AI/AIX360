from unittest import TestCase

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aix360.algorithms.rbm import FeatureBinarizerFromTrees


# noinspection PyPep8Naming
class TestFeatureBinarizerFromTrees(TestCase):

    def setUp(self) -> None:
        self.random_state = 0
        d: dict = load_breast_cancer()
        X: DataFrame = DataFrame(d['data'], columns=d['feature_names'])
        self.col_ordinal = X.columns.to_list()
        np.random.seed(self.random_state)
        s = np.array(['a', 'b', 'c'])
        X['cat alpha'] = s[np.random.randint(0, 3, len(X))]
        X['cat num'] = np.random.randint(0, 3, len(X))
        self.col_categorical = ['cat alpha', 'cat num']
        s = np.array(['a', 'b'])
        X['bin alpha'] = s[np.random.randint(0, 2, len(X))]
        X['bin num'] = np.random.randint(0, 2, len(X))
        self.col_binary = ['bin alpha', 'bin num']
        self.X = X
        self.y: ndarray = d['target']
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.4, random_state=self.random_state)

    def test_init(self):

        # colCateg >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        fbt = FeatureBinarizerFromTrees(colCateg=self.col_categorical)
        self.assertListEqual(fbt.colCateg, self.col_categorical)

        # treeNum  >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        with self.assertRaises(ValueError):
            fbt = FeatureBinarizerFromTrees(treeNum=None)
            
        with self.assertRaises(ValueError):
            fbt = FeatureBinarizerFromTrees(treeNum=0)
            
        with self.assertRaises(ValueError):
            fbt = FeatureBinarizerFromTrees(treeNum=-1)

        fbt = FeatureBinarizerFromTrees(treeNum=3)
        self.assertEqual(fbt.treeNum, 3)

        # treeDepth >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        with self.assertRaises(ValueError):
            fbt = FeatureBinarizerFromTrees(treeDepth=0)
            
        with self.assertRaises(ValueError):
            fbt = FeatureBinarizerFromTrees(treeDepth=-1)

        fbt = FeatureBinarizerFromTrees(treeDepth=3)
        self.assertEqual(fbt.treeDepth, 3)
        self.assertEqual(fbt.treeKwargs['max_depth'], 3)

        fbt = FeatureBinarizerFromTrees(treeDepth=None, treeKwargs=dict(max_depth=5))
        self.assertEqual(fbt.treeKwargs['max_depth'], 5)
        self.assertEqual(fbt.treeDepth, 5)

        # treeFeatureSelection  >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        with self.assertRaises(ValueError):
            fbt = FeatureBinarizerFromTrees(treeFeatureSelection=0)
            
        with self.assertRaises(ValueError):
            fbt = FeatureBinarizerFromTrees(treeFeatureSelection=-1)

        with self.assertRaises(ValueError):
            fbt = FeatureBinarizerFromTrees(treeFeatureSelection=3)

        with self.assertRaises(ValueError):
            fbt = FeatureBinarizerFromTrees(treeFeatureSelection='bad string value')

        fbt = FeatureBinarizerFromTrees(treeFeatureSelection=0.4)
        self.assertEqual(fbt.treeFeatureSelection, 0.4)
        self.assertEqual(fbt.treeKwargs['max_features'], 0.4)

        fbt = FeatureBinarizerFromTrees(treeFeatureSelection=None)
        self.assertTrue(fbt.treeFeatureSelection is None)
        self.assertTrue(fbt.treeKwargs['max_features'] is None)

        fbt = FeatureBinarizerFromTrees(treeFeatureSelection='log2')
        self.assertEqual(fbt.treeFeatureSelection, 'log2')
        self.assertEqual(fbt.treeKwargs['max_features'], 'log2')

        fbt = FeatureBinarizerFromTrees(treeFeatureSelection=None, treeKwargs=dict(max_features=0.2))
        self.assertEqual(fbt.treeKwargs['max_features'], 0.2)
        self.assertEqual(fbt.treeFeatureSelection, 0.2)

        # threshRound >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        fbt = FeatureBinarizerFromTrees(threshRound=None)

        with self.assertRaises(ValueError):
            FeatureBinarizerFromTrees(threshRound=-1)

        fbt = FeatureBinarizerFromTrees(threshRound=3)
        self.assertTrue(fbt.threshRound == 3)

        # threshStr > >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        fbt = FeatureBinarizerFromTrees(threshStr=True)
        self.assertTrue(fbt.threshStr)
        fbt = FeatureBinarizerFromTrees(threshStr=False)
        self.assertFalse(fbt.threshStr)

        # returnOrd   >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        fbt = FeatureBinarizerFromTrees(returnOrd=True)
        self.assertTrue(fbt.returnOrd)
        fbt = FeatureBinarizerFromTrees(returnOrd=False)
        self.assertFalse(fbt.returnOrd)

        # randomState >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        fbt = FeatureBinarizerFromTrees(randomState=3)
        self.assertEqual(fbt.randomState, 3)

    def test_fit_and_transform_exceptions(self):

        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        # fit() requires y. The error is raised at 'self.decisionTree.fit(X, y)'
        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>

        fbt = FeatureBinarizerFromTrees()
        with self.assertRaises((TypeError, ValueError)):
            fbt.fit(self.X_train)

        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        # fit() does not allow/support NaN/None
        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>

        fbt = FeatureBinarizerFromTrees()
        Xn = self.X_train.copy(True)
        Xn.iloc[[31, 27, 80], Xn.columns.get_loc('smoothness error')] = np.NaN
        with self.assertRaises(ValueError):
            fbt.fit(Xn, self.y_train)

        fbt = FeatureBinarizerFromTrees()
        Xn = self.X_train.copy(True)
        Xn.iloc[[3, 17, 20], Xn.columns.get_loc('bin num')] = np.NaN
        with self.assertRaises(ValueError):
            fbt.fit(Xn, self.y_train)

        fbt = FeatureBinarizerFromTrees(colCateg=self.col_categorical)
        Xn = self.X_train.copy(True)
        Xn.iloc[[3, 17, 20], Xn.columns.get_loc('cat num')] = np.NaN
        with self.assertRaises(ValueError):
            fbt.fit(Xn, self.y_train)

    def test_fit_and_transform_binary(self):

        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        # Test binary features with no categorical or ordinal features.
        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        fbt = FeatureBinarizerFromTrees(treeNum=1, treeDepth=1, randomState=self.random_state)
        fbt.fit(self.X_train[self.col_binary], self.y_train)
        self.assertListEqual(list(fbt.maps.keys()), ['bin alpha'])
        temp = [('bin alpha', '', ''), ('bin alpha', 'not', '')]
        self.assertListEqual(fbt.features.to_list(), temp)

        # Transform
        T = fbt.transform(self.X_test)
        self.assertListEqual(T.columns.to_list(), temp)

        # Now test taking all available features.
        fbt = FeatureBinarizerFromTrees(treeNum=1, treeDepth=None, randomState=self.random_state)
        fbt.fit(self.X_train[self.col_binary], self.y_train)
        self.assertListEqual(list(fbt.maps.keys()), self.col_binary)
        temp = [('bin alpha', '', ''), ('bin alpha', 'not', ''), ('bin num', '', ''), ('bin num', 'not', '')]
        self.assertListEqual(fbt.features.to_list(), temp)

        # Transform
        T = fbt.transform(self.X_test)
        self.assertListEqual(fbt.features.to_list(), temp)
        a = T[('bin num', '', '')].to_numpy()
        b = (self.X_test['bin num'] == 1).astype(int).to_numpy()
        self.assertTrue(np.all(a == b))
        a = T[('bin num', 'not', '')].to_numpy()
        b = 1 - b
        self.assertTrue(np.all(a == b))
        a = T[('bin alpha', '', '')].to_numpy()
        b = (self.X_test['bin alpha'] == 'b').astype(int).to_numpy()
        self.assertTrue(np.all(a == b))
        a = T[('bin alpha', 'not', '')].to_numpy()
        b = 1 - b
        self.assertTrue(np.all(a == b))

    def test_fit_and_transform_categorical(self):

        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        # Test categorical with no binary or ordinal features.
        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>

        # Two features (one feature with == and !=).
        fbt = FeatureBinarizerFromTrees(treeNum=1, treeDepth=1, colCateg=self.col_categorical,
                                        randomState=self.random_state)
        fbt.fit(self.X_train[self.col_categorical], self.y_train)
        self.assertListEqual(list(fbt.enc.keys()), [self.col_categorical[1]])
        self.assertTrue(type(list(fbt.enc.values())[0]) is OneHotEncoder)
        temp = [('cat num', '!=', 0), ('cat num', '==', 0)]
        self.assertListEqual(fbt.features.to_list(), temp)

        # Test transform. Categorical values are converted to strings to be like FeatureBinarizer
        T = fbt.transform(self.X_test)
        self.assertListEqual(T.columns.to_list(), temp)

        # Test taking all available features.
        fbt = FeatureBinarizerFromTrees(treeNum=1, treeDepth=None, colCateg=self.col_categorical,
                                        randomState=self.random_state)
        fbt.fit(self.X_train[self.col_categorical], self.y_train)
        self.assertListEqual(self.col_categorical, fbt.colCateg)
        self.assertListEqual(self.col_categorical, list(fbt.enc.keys()))
        temp = [('cat alpha', '!=', 'a'), ('cat alpha', '!=', 'c'), ('cat alpha', '==', 'a'), ('cat alpha', '==', 'c'),
                ('cat num', '!=', 0), ('cat num', '!=', 2), ('cat num', '==', 0), ('cat num', '==', 2)]
        self.assertListEqual(fbt.features.to_list(), temp)

        # Transform
        T = fbt.transform(self.X_test)
        self.assertListEqual(T.columns.to_list(), temp)
        a = T[('cat alpha', '==', 'a')].to_numpy()
        b = (self.X_test['cat alpha'] == 'a').astype(int).to_numpy()
        self.assertTrue(np.all(a == b))
        a = T[('cat alpha', '!=', 'a')].to_numpy()
        b = 1 - b
        self.assertTrue(np.all(a == b))
        a = T[('cat num', '==', 2)].to_numpy()
        b = (self.X_test['cat num'] == 2).astype(int).to_numpy()
        self.assertTrue(np.all(a == b))
        a = T[('cat num', '!=', 2)].to_numpy()
        b = 1 - b
        self.assertTrue(np.all(a == b))

    def test_fit_and_transform_ordinal(self):

        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        # Test ordinal with no categorical or binary features.
        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>

        fbt = FeatureBinarizerFromTrees(treeNum=1, treeDepth=1, randomState=self.random_state)
        fbt.fit(self.X_train[self.col_ordinal], self.y_train)
        temp = [('mean concave points', '<=', 0.04892), ('mean concave points',  '>', 0.04892)]
        self.assertListEqual(fbt.features.to_list(), temp)
        self.assertDictEqual(fbt.thresh, {'mean concave points': np.array([0.04892])})

        # Transform
        T = fbt.transform(self.X_test)
        self.assertListEqual(T.columns.to_list(), temp)

        # Test threshStr
        fbt = FeatureBinarizerFromTrees(treeNum=1, treeDepth=1, randomState=self.random_state, threshStr=True)
        fbt.fit(self.X_train[self.col_ordinal], self.y_train)
        self.assertDictEqual(fbt.thresh, {'mean concave points': np.array([0.04892])})

        # Transform
        T = fbt.transform(self.X_test)
        temp = [('mean concave points', '<=', '0.04892'), ('mean concave points',  '>', '0.04892')]
        self.assertListEqual(T.columns.to_list(), temp)

        # Test threshRound
        fbt = FeatureBinarizerFromTrees(treeNum=1, treeDepth=1, threshRound=2, randomState=self.random_state)
        fbt.fit(self.X_train[self.col_ordinal], self.y_train)
        temp = [('mean concave points', '<=', 0.05), ('mean concave points',  '>', 0.05)]
        self.assertListEqual(fbt.features.to_list(), temp)
        self.assertDictEqual(fbt.thresh, {'mean concave points': np.array([0.05])})

        # Now test taking all available features.
        fbt = FeatureBinarizerFromTrees(treeNum=1, treeDepth=None, randomState=self.random_state)
        fbt.fit(self.X_train[self.col_ordinal], self.y_train)

        temp = {'area error': np.array([46.315001]), 'concavity error': np.array([0.016965]),
                'mean area': np.array([995.5]), 'mean concave points': np.array([0.04892]),
                'mean texture': np.array([19.9]), 'smoothness error': np.array([0.003299, 0.005083]),
                'texture error': np.array([0.51965]), 'worst area': np.array([785.799988]),
                'worst compactness': np.array([0.4508]), 'worst concavity': np.array([0.3655]),
                'worst texture': np.array([23.47, 32.779999, 33.805])}

        for k, v in fbt.thresh.items():
            self.assertTrue(np.all(temp[k] == v))

        temp = [('area error', '<=', 46.315001), ('area error', '>', 46.315001), ('concavity error', '<=', 0.016965),
                ('concavity error', '>', 0.016965), ('mean area', '<=', 995.5), ('mean area', '>', 995.5),
                ('mean concave points', '<=', 0.04892), ('mean concave points', '>', 0.04892),
                ('mean texture', '<=', 19.9), ('mean texture', '>', 19.9), ('smoothness error', '<=', 0.003299),
                ('smoothness error', '<=', 0.005083), ('smoothness error', '>', 0.003299),
                ('smoothness error', '>', 0.005083), ('texture error', '<=', 0.51965), ('texture error', '>', 0.51965),
                ('worst area', '<=', 785.799988), ('worst area', '>', 785.799988), ('worst compactness', '<=', 0.4508),
                ('worst compactness', '>', 0.4508), ('worst concavity', '<=', 0.3655), ('worst concavity', '>', 0.3655),
                ('worst texture', '<=', 23.47), ('worst texture', '<=', 32.779999), ('worst texture', '<=', 33.805),
                ('worst texture', '>', 23.47), ('worst texture', '>', 32.779999), ('worst texture', '>', 33.805)]

        self.assertListEqual(fbt.features.to_list(), temp)

    def test_fit_and_transform_return_ordinal(self):

        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        # returnOrd
        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>

        fbt = FeatureBinarizerFromTrees(treeNum=1, treeDepth=3, returnOrd=True, colCateg=self.col_categorical,
                                        randomState=self.random_state)
        fbt.fit(self.X_train[['mean area', 'mean concave points', 'cat num', 'cat alpha']], self.y_train)
        self.assertListEqual(fbt.ordinal, ['mean area', 'mean concave points'])
        self.assertTrue(type(fbt.scaler) is StandardScaler)
        self.assertTrue(fbt.scaler.transform(self.X_test[fbt.ordinal]).shape[1] == 2)

    def test_fit_and_transform_all_feature_classes(self):

        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        # All feature classes together
        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>

        fbt = FeatureBinarizerFromTrees(
            colCateg=self.col_categorical,
            treeNum=1,
            treeDepth=None,
            threshRound=2,
            returnOrd=True,
            randomState=self.random_state
        )

        # Reduce the number of numerical features, otherwise the categorical and binary features won't be selected.
        cols = self.col_categorical + self.col_binary + self.col_ordinal[0:2]
        fbt.fit(self.X_train[cols], self.y_train)
        T = fbt.transform(self.X_test[cols])
        self.assertTrue(type(T) is tuple)
        self.assertTrue(type(T[0]) is DataFrame)
        self.assertTrue(type(T[1]) is DataFrame)
        U: DataFrame = T[1]
        T: DataFrame = T[0]
        self.assertListEqual(U.columns.to_list(), fbt.ordinal)

        self.assertListEqual(list(fbt.enc.keys()), self.col_categorical)
        self.assertTrue(type(list(fbt.enc.values())[0]) is OneHotEncoder)
        self.assertTrue(type(list(fbt.thresh.values())[0]) is ndarray)
        self.assertListEqual(list(fbt.maps.keys()), self.col_binary)
        self.assertListEqual(list(fbt.thresh.keys()), ['mean radius', 'mean texture'])

        a = T[('mean texture', '<=', 27.24)].to_numpy()
        b = (self.X_test['mean texture'] <= 27.24).astype(int).to_numpy()
        self.assertTrue(np.all(a == b))
        a = T[('mean texture', '>', 27.24)].to_numpy()
        b = 1 - b
        self.assertTrue(np.all(a == b))

        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>
        # Test NaN, None during transform.
        #
        # In the past, Numpy raised a warning when doing vector comparisons against Nan/None,
        # but this is no longer the case.
        # >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >> >>

        Xn: DataFrame = self.X_test.copy(True)
        idx = Xn.sample(10).index
        Xn.loc[idx, 'mean radius'] = np.NaN
        Xn.loc[idx, 'cat alpha'] = None
        T: DataFrame = fbt.transform(Xn)[0]
        # For continuous values, the NaN values do not qualify for any range.
        self.assertFalse((T.loc[idx, 'mean radius'] == 1).to_numpy().any())
        # For categorical values, the None values do not qualify for any comparisons.
        self.assertTrue((T.loc[idx, ('cat alpha', '==')] == 0).to_numpy().all())
        self.assertTrue((T.loc[idx, ('cat alpha', '!=')] == 1).to_numpy().all())
