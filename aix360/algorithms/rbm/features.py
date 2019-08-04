import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureBinarizer(TransformerMixin):
    '''Transformer for binarizing categorical and ordinal (including continuous) features.
    
    For use with LogisticRuleRegression and LinearRuleRegression
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
                maps[c] = pd.Series(range(valUniq), index=np.sort(data[c].unique()))

            # Categorical column
            elif (c in self.colCateg) or (data[c].dtype == 'object'):
                # OneHotEncoder object
                enc[c] = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')
                # Fit to observed categories
                enc[c].fit(data[[c]])

            # Ordinal column
            elif np.issubdtype(data[c].dtype, np.dtype(int).type) \
                | np.issubdtype(data[c].dtype, np.dtype(float).type):
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
                A[(str(c), '', '')] = data[c].map(maps[c])
                if self.negations:
                    A[(str(c), 'not', '')] = 1 - A[(str(c), '', '')]

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
