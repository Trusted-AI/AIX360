from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# noinspection PyPep8Naming
class FeatureBinarizer(TransformerMixin):
    '''Transformer for binarizing categorical and ordinal features.
    
    For use with BooleanRuleCG, LogisticRuleRegression and LinearRuleRegression
    '''
    def __init__(self, colCateg=[], numThresh=9, negations=False, threshStr=False, returnOrd=False, **kwargs):
        """
        Args:
            colCateg (list): Categorical features ('object' dtype automatically treated as categorical)
            numThresh (int): Number of quantile thresholds used to binarize ordinal variables
            negations (bool): Append negations
            threshStr (bool): Convert thresholds on ordinal features to strings
            returnOrd (bool): Also return standardized ordinal features
        """
        # List of categorical columns
        if type(colCateg) is pd.Series:
            self.colCateg = colCateg.tolist()
        elif type(colCateg) is not list:
            self.colCateg = [colCateg]
        else:
            self.colCateg = colCateg
        # Number of quantile thresholds used to binarize ordinal features
        self.numThresh = numThresh
        self.thresh = {}
        # whether to append negations
        self.negations = negations
        # whether to convert thresholds on ordinal features to strings
        self.threshStr = threshStr
        # Also return standardized ordinal features
        self.returnOrd = returnOrd

    def fit(self, X):
        '''Fit FeatureBinarizer to data
        
        Args:
            X (DataFrame): Original features
        Returns:
            FeatureBinarizer: Self
            self.maps (dict): Mappings for unary/binary columns
            self.enc (dict): OneHotEncoders for categorical columns
            self.thresh (dict(array)): Thresholds for ordinal columns
            self.NaN (list): Ordinal columns containing NaN values
            self.ordinal (list): Ordinal columns
            self.scaler (StandardScaler): StandardScaler for ordinal columns
        '''
        data = X
        # Quantile probabilities
        quantProb = np.linspace(1. / (self.numThresh + 1.), self.numThresh / (self.numThresh + 1.), self.numThresh)
        # Initialize
        maps = {}
        enc = {}
        thresh = {}
        NaN = []
        if self.returnOrd:
            ordinal = []

        # Iterate over columns
        for c in data:
            # number of unique values
            valUniq = data[c].nunique()

            # Constant or binary column
            if valUniq <= 2:
                # Mapping to 0, 1
                maps[c] = pd.Series(range(valUniq), index=np.sort(data[c].dropna().unique()))

            # Categorical column
            elif (c in self.colCateg) or (data[c].dtype == 'object'):
                # OneHotEncoder object
                enc[c] = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')
                # Fit to observed categories
                enc[c].fit(data[[c]])

            # Ordinal column
            elif np.issubdtype(data[c].dtype, np.integer) | np.issubdtype(data[c].dtype, np.floating):
                # Few unique values
                if valUniq <= self.numThresh + 1:
                    # Thresholds are sorted unique values excluding maximum
                    thresh[c] = np.sort(data[c].unique())[:-1]
                # Many unique values
                else:
                    # Thresholds are quantiles excluding repetitions
                    thresh[c] = data[c].quantile(q=quantProb).unique()
                if data[c].isnull().any():
                    # Contains NaN values
                    NaN.append(c)
                if self.returnOrd:
                    ordinal.append(c)

            else:
                print(("Skipping column '" + str(c) + "': data type cannot be handled"))
                continue

        self.maps = maps
        self.enc = enc
        self.thresh = thresh
        self.NaN = NaN
        if self.returnOrd:
            self.ordinal = ordinal
            # Fit StandardScaler to ordinal features
            self.scaler = StandardScaler().fit(data[ordinal])
        return self

    def transform(self, X):
        '''Binarize features
        
        Args:
            X (DataFrame): Original features
        Returns:
            A (DataFrame): Binarized features with MultiIndex column labels
            Xstd (DataFrame, optional): Standardized ordinal features
        '''
        data = X
        maps = self.maps
        enc = self.enc
        thresh = self.thresh
        NaN = self.NaN

        # Initialize dataframe
        A = pd.DataFrame(index=data.index,
                         columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))

        # Iterate over columns
        for c in data:
            # Constant or binary column
            if c in maps:
                # Rename values to 0, 1
                colName = (str(c), '!=', str(maps[c].index[0])) if len(maps[c]) == 1 else (str(c), '==', str(maps[c].index[1]))
                A[colName] = data[c].map(maps[c]).astype(int)
                if self.negations:
                    A[(str(c), '==', str(maps[c].index[0]))] = 1 - A[colName]

            # Categorical column
            elif c in enc:
                # Apply OneHotEncoder
                Anew = enc[c].transform(data[[c]])
                Anew = pd.DataFrame(Anew, index=data.index, columns=enc[c].categories_[0].astype(str))
                if self.negations:
                    # Append negations
                    Anew = pd.concat([Anew, 1 - Anew], axis=1, keys=[(str(c), '=='), (str(c), '!=')])
                else:
                    Anew.columns = pd.MultiIndex.from_product([[str(c)], ['=='], Anew.columns])
                # Concatenate
                A = pd.concat([A, Anew], axis=1)

            # Ordinal column
            elif c in thresh:
                # Threshold values to produce binary arrays
                Anew = (data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
                if self.negations:
                    # Append negations
                    Anew = np.concatenate((Anew, 1 - Anew), axis=1)
                    ops = ['<=', '>']
                else:
                    ops = ['<=']
                # Convert to dataframe with column labels
                if self.threshStr:
                    Anew = pd.DataFrame(Anew, index=data.index,
                                        columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c].astype(str)]))
                else:
                    Anew = pd.DataFrame(Anew, index=data.index,
                                        columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c]]))
                if c in NaN:
                    # Ensure that rows corresponding to NaN values are zeroed out
                    indNull = data[c].isnull()
                    Anew.loc[indNull] = 0
                    # Add NaN indicator column
                    Anew[(str(c), '==', 'NaN')] = indNull.astype(int)
                    if self.negations:
                        Anew[(str(c), '!=', 'NaN')] = (~indNull).astype(int)
                # Concatenate
                A = pd.concat([A, Anew], axis=1)

            else:
                print(("Skipping column '" + str(c) + "': data type cannot be handled"))
                continue

        if self.returnOrd:
            # Standardize ordinal features
            Xstd = self.scaler.transform(data[self.ordinal])
            Xstd = pd.DataFrame(Xstd, index=data.index, columns=self.ordinal)
            # Fill NaN with mean (which is now zero)
            Xstd.fillna(0, inplace=True)
            return A, Xstd
        else:
            return A


# noinspection PyPep8Naming
class FeatureBinarizerFromTrees(TransformerMixin):
    """Transformer for binarizing categorical and ordinal features.

    For use with BooleanRuleCG, LogisticRuleRegression, and LinearRuleRegression. This transformer generates binary
    features using splits in decision trees. Compared to `FeatureBinarizer`, this approach reduces the number of
    features required to produce an accurate model. The smaller feature space shortens training time and often
    simplifies rule sets.
    """

    def __init__(self,
                 colCateg: list = None,
                 treeNum: int = 1,
                 treeDepth: Optional[int] = 4,
                 treeFeatureSelection: Union[str, float, None] = None,
                 treeKwargs: dict = None,
                 threshRound: Optional[int] = 6,
                 threshStr: bool = False,
                 returnOrd: bool = False,
                 randomState: int = None,
                 **kwargs):
        """
        Args:
            colCateg (list): Categorical features ('object' dtype automatically treated as categorical). These features
                are one-hot-encoded.
            treeNum (int): Number of trees to fit. Setting 'treeNum' to a value greater than one usually produces a
                larger variety of output features.
            treeDepth (int): The maximum depth of the tree. Setting 'treeDepth=None' grows a tree without limit.
                Larger depth values produce more output features. Corresponds to parameter 'max_depth' in
                DecisionTreeClassifier.
            treeFeatureSelection (float, str): When building a tree, the input features are randomly permuted at
                each split. This parameter specifies how many input features are considered at each split. By default,
                this parameter is set to 'None' which indicates that all features should be considered at every split.
                Other possible values are 'sqrt', 'log2', or a float that indicates the proportion of features to
                select at every split (e.g. 0.5 would randomly select half of the input features at every split).
                To create a wide variety of output features, or to sift through a very large number of features,
                increase 'treeNum' and set 'treeFeatureSelection="sqrt"'. Corresponds to 'max_features' in
                DecisionTreeClassifier.
            treeKwargs (dict): A dictionary of parameters to pass to the scikit-learn DecisionTreeClassifier during
                fitting.
            threshRound (int): Round threshold values by this number of decimal places. This parameter can be used
                to prevent similar thresholds from generating separate binarized features. E.g., if 'threshRound=2',
                only one binarized feature will be generated for thresholds 0.009 and 0.01. Setting 'threshRound=None'
                will disable rounding.
            threshStr (bool): Convert threshold values to strings, including categorical values, in transformed
                data frame's index.
            returnOrd (bool): Return a standardized data frame for ordinal features (both discrete and continuous)
                during transformation in addition to the binarized data frame.
            randomState (int): Random state for decision tree.
        """

        # Categorical columns
        if colCateg is None:
            self.colCateg = []
        elif type(colCateg) is Series:
            self.colCateg = colCateg.to_list()
        elif type(colCateg) is not list:
            self.colCateg = [colCateg]
        else:
            self.colCateg = colCateg

        # Number of trees
        if (treeNum is None) or (treeNum < 1) or (int(treeNum) != treeNum):
            raise ValueError('The value for \'treeNum\' must be an integer value greater than zero.')
        self.treeNum = int(treeNum)

        # Tree kwargs
        if treeKwargs is None:
            treeKwargs = dict(max_features=None)
        elif 'max_features' not in treeKwargs:
            treeKwargs['max_features'] = None

        # Tree depth
        if treeDepth is not None:
            if (treeDepth < 1) or (int(treeDepth) != treeDepth):
                raise ValueError('The value for \'treeDepth\' must be None or an integer value greater than zero.')
            treeKwargs['max_depth'] = treeDepth
        elif 'max_depth' in treeKwargs:
            treeDepth = treeKwargs['max_depth']
        self.treeDepth = treeDepth

        # Tree feature selection
        if treeFeatureSelection is not None:
            if isinstance(treeFeatureSelection, str):
                error = treeFeatureSelection not in ('auto', 'sqrt', 'log2')
            elif isinstance(treeFeatureSelection, (float, int)):
                error = (treeFeatureSelection <= 0.0) or (treeFeatureSelection > 1.0)
            else:
                error = True
            if error:
                raise ValueError('Valid values for \'treeFeatureSelection\' are None, \'auto\', \'sqrt\', \'log2\', or '
                                 'a value in interval (0, 1].')
            treeKwargs['max_features'] = treeFeatureSelection
        elif 'max_features' in treeKwargs:
            treeFeatureSelection = treeKwargs['max_features']
        self.treeFeatureSelection = treeFeatureSelection

        # Tree kwargs
        self.treeKwargs = treeKwargs

        # Random state
        self.randomState = randomState

        # Rounding for ordinal values
        if (threshRound is not None) and (threshRound < 0):
            raise ValueError('The value for \'threshRound\' must be None, or zero, or greater than zero.')
        self.threshRound = threshRound

        # Whether to convert thresholds on ordinal features to strings
        self.threshStr = threshStr

        # Whether to convert thresholds on ordinal features to strings.  Also return standardized ordinal features
        # during transformation
        self.returnOrd = returnOrd

    def _fit_transform_like_feature_binarizer(self, X: DataFrame) -> DataFrame:
        # Initialize
        maps = {}
        enc = {}
        thresh = {}
        ordinal = []
        A = DataFrame(index=X.index,
                      columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))

        # Iterate over columns
        for c in X:
            # number of unique values
            valUniq = X[c].nunique()

            # Constant or binary column
            if valUniq <= 2:
                # Mapping to 0, 1
                maps[c] = pd.Series(range(valUniq), index=np.sort(X[c].dropna().unique()))
                A[(str(c), '', '')] = X[c].map(maps[c])

            # Categorical column
            elif (c in self.colCateg) or (X[c].dtype == 'object'):
                # In the past, OneHotEncoder did not support NaN. Now it does, but it doesn't logically work for this
                # scenario. Check for NaNs and throw the same exception manually.
                if X[[c]].isna().any()[0]:
                    raise ValueError('Categorical input contains NaN.')
                # OneHotEncoder object
                enc[c] = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')
                # Fit to observed categories
                enc[c].fit(X[[c]])
                # Apply OneHotEncoder
                Anew = enc[c].transform(X[[c]])
                # Original FeatureBinarizer converts all values to str. This class preserves type to be used
                # during transform.
                Anew = DataFrame(Anew, index=X.index, columns=enc[c].categories_[0])
                Anew.columns = pd.MultiIndex.from_product([[str(c)], ['=='], Anew.columns])
                # Concatenate
                A = pd.concat([A, Anew], axis=1)

            # Ordinal column
            elif np.issubdtype(X[c].dtype, np.integer) | np.issubdtype(X[c].dtype, np.floating):
                # Unlike FeaturBinarizer, just append the original ordinal column. It will be fit by the
                # DecisionTreeClassifier.
                Anew = DataFrame(
                    X[c].to_numpy(),  # Required
                    columns=pd.MultiIndex.from_arrays([[c], ['<='], [0.0]], names=['feature', 'operation', 'value']),
                    index=X.index
                )
                A = pd.concat([A, Anew], axis=1)
                ordinal.append(c)

            else:
                print(("Skipping column '" + str(c) + "': data type cannot be handled"))
                continue

        self.maps = maps
        self.enc = enc
        self.thresh = thresh
        self.ordinal = ordinal

        return A

    def fit(self, X: DataFrame, y: Union[ndarray, DataFrame, Series, list] = None):
        """Fit transformer. NaN/None values are not permitted for X or y.

        Args:
            X (DataFrame): Original features
            y (Iterable): Target
        Returns:
            FeatureBinarizerFromTrees: Self
            self.enc (dict): OneHotEncoders for categorical columns
            self.features (MultiIndex): Pandas MultiIndex of feature names, operations, and values
            self.maps (dict): Mappings for unary/binary columns
            self.ordinal (list): Ordinal columns
            self.scaler (StandardScaler): StandardScaler for ordinal columns
            self.thresh (dict(array)): Thresholds for ordinal columns
        """

        # The decision tree will also throw an exception, but it is cryptic.
        if y is None:
            raise ValueError('The parameter \'y\' is required.')

        # Binarize unary/binary/categorical features according to the FeatureBinarizer style. Ordinal columns
        # are not binarized: i.e., they are included as-is in the data frame with '<=' operations in the index.
        # They will be binarized by extracting splits from the decision tree.
        Xfb = self._fit_transform_like_feature_binarizer(X)

        # Save resulting MultiIndex separately and reset columns. scikit-learn will not support them in version 1.2
        # during fit().
        XfbMultiIndex = Xfb.columns.copy()
        Xfb.columns = range(0, Xfb.shape[1])

        # Fit decision trees.
        featuresIdx = np.empty((0,), int)
        thresholds = np.empty((0,), int)

        randomState = self.randomState
        for i in range(self.treeNum):
            if randomState is not None:
                randomState += i
            tree = DecisionTreeClassifier(random_state=randomState, **self.treeKwargs)
            tree.fit(Xfb, y)
            featuresIdx = np.hstack((featuresIdx, tree.tree_.feature))
            thresholds = np.hstack((thresholds, tree.tree_.threshold))

        # See `Understanding the decision tree structure`
        # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        select = featuresIdx != -2  # Indicates leaf nodes
        featuresIdx = featuresIdx[select]
        thresholds = thresholds[select]

        # Important to do rounding before dropping duplicates because rounding can create duplicates.
        if (self.threshRound is not None) and len(self.ordinal):
            thresholds = thresholds.round(self.threshRound)

        # Create frame from column index which contains the relevant features
        features: DataFrame = XfbMultiIndex[featuresIdx].to_frame(False)

        # Set thresholds for ordinal values
        if len(self.ordinal):
            select = (features['operation'] == '<=').to_numpy()
            features.loc[select, 'value'] = thresholds[select]

        # Drop duplicate features.
        features.drop_duplicates(inplace=True)

        # Create rule pairs for each feature
        temp: DataFrame = features.copy(True)
        temp['operation'].replace({'<=': '>', '==': '!=', '': 'not'}, inplace=True)
        features = pd.concat((features, temp))

        # Create/sort multi-index that will be used during transformation.
        self.features = pd.MultiIndex.from_frame(features)
        self.features = self.features.sortlevel((0, 1, 2))[0]

        # Update respective self attributes based on the selected features.
        if '==' in self.features.levels[1]:
            names = self.features.get_loc_level('==', 'operation')[1].get_level_values('feature').unique()
            self.enc = {k: self.enc[k] for k in names}
        else:
            self.enc = {}

        if '' in self.features.levels[1]:
            names = self.features.get_loc_level('', 'operation')[1].get_level_values('feature').unique()
            self.maps = {k: self.maps[k] for k in names}
        else:
            self.maps = {}

        if '<=' in self.features.levels[1]:
            names = self.features.get_loc_level('<=', 'operation')[1].get_level_values('feature').unique()
            self.thresh = \
                {k: self.features
                    .get_loc_level([k, '<='], ['feature', 'operation'])[1]
                    .get_level_values('value')
                    .to_numpy(dtype=float)
                 for k in names}
            self.ordinal = names.to_list()
            if self.returnOrd:
                self.scaler = StandardScaler().fit(X[names])
        else:
            self.thresh = {}
            self.ordinal = []
            self.scaler = None

        return self

    def transform(self, X: DataFrame) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        """Binarize features. Binary features are sorted name-operation-value.

        Args:
            X (DataFrame): Original features
        Returns:
            A (DataFrame): Binarized features with MultiIndex column labels
            Xstd (DataFrame, optional): Standardized ordinal features
        """

        result: DataFrame = DataFrame(
            np.zeros((X.shape[0], len(self.features)), dtype=int),  # Type consistent with original FeatureBinarizer
            columns=self.features,
            index=X.index
        )

        # Using to_numpy() speeds up the overall transform by more than 2x because indices aren't created/aligned.
        # Futhermore, using numpy raises warnings for NaN/None values which is what we want here.
        for feature, test, value in result.columns:
            if test == '<=':
                v = X[feature].to_numpy() <= value
            elif test == '>':
                v = X[feature].to_numpy() > value
            elif test == '==':
                v = X[feature].to_numpy() == value
            elif test == '!=':
                v = X[feature].to_numpy() != value
            elif test == '':
                v = X[feature].to_numpy() == self.maps[feature].index[1]
            elif test == 'not':
                v = X[feature].to_numpy() == self.maps[feature].index[0]
            else:
                raise RuntimeError(f'Test operation \'{test}\' not supported.')

            # Faster to replace column with numpy because there is no index to align/join
            result[(feature, test, value)] = v.astype(int)

        if self.threshStr:
            result.columns = result.columns.set_levels(levels=result.columns.levels[2].astype(str), level='value')

        # This is taken from FeatureBinarizer.
        if self.returnOrd:
            # Standardize ordinal features
            Xstd = self.scaler.transform(X[self.ordinal])
            Xstd = DataFrame(Xstd, index=X.index, columns=self.ordinal)
            return result, Xstd
        else:
            return result
