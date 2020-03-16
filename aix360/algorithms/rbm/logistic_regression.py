import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from .beam_search import beam_search_K1


class LogisticRuleRegression(BaseEstimator, ClassifierMixin):
    """Logistic Rule Regression is a directly interpretable supervised learning
    method that performs logistic regression on rule-based features.
    """
    def __init__(self,
      lambda0=0.05,
      lambda1=0.01,
      useOrd=False,
      debias=True,
      init0=False,
      K=1,
      iterMax=200,
      B=1,
      wLB=0.5,
      stopEarly=False,
      eps=1e-6,
      maxSolverIter=100):

        """
        Args:
            lambda0 (float, optional): Regularization - fixed cost of each rule
            lambda1 (float, optional): Regularization - additional cost of each literal in rule
            useOrd (bool, optional): Also use standardized numerical features
            debias (bool, optional): Re-fit final solution without regularization
            init0 (bool, optional): Initialize with no features
            K (int, optional): Column generation - maximum number of columns generated per iteration
            iterMax (int, optional): Column generation - maximum number of iterations
            B (int, optional): Column generation - beam search width
            wLB (float, optional): Column generation - weight on lower bound in evaluating nodes
            stopEarly (bool, optional): Column generation - stop after current degree once improving column found
            eps (float, optional): Numerical tolerance on comparisons
            maxSolverIter: Maximum number of logistic regression solver iterations
        """
        # Use standardized ordinal features
        self.useOrd = useOrd
        # Initialize with no features
        self.init0 = init0
        # Regularization parameters
        self.lambda0 = lambda0      # fixed cost of term
        self.lambda1 = lambda1      # cost per literal
        self.debias = debias        # re-fit final solution without regularization
        # Column generation parameters
        self.K = K                  # maximum number of columns generated per iteration
        self.iterMax = iterMax      # maximum number of iterations
        self.B = B                  # beam search width
        self.wLB = wLB              # weight on lower bound in evaluating nodes
        self.stopEarly = stopEarly  # stop after current degree once improving column found
        # Numerical tolerance on comparisons
        self.eps = eps
        # Maximum logistic solver iterations
        self.maxSolverIter = maxSolverIter


    def fit(self, X, y, Xstd=None):
        """Fit model to training data.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
            y (array): Target variable
            Xstd (DataFrame, optional): Standardized numerical features
        Returns:
            LogisticRuleRegression: Self
        """
        # Initialization
        # Number of samples
        n = X.shape[0]
        if self.init0:
            # Initialize with empty feature indicator and conjunction matrices
            z = pd.DataFrame([], index=X.columns)
            A = np.empty((X.shape[0], 0))
        else:
            # Initialize with X itself i.e. singleton conjunctions
            # Feature indicator and conjunction matrices
            z = pd.DataFrame(np.eye(X.shape[1], dtype=int), index=X.columns)
            # Remove negations
            indPos = X.columns.get_level_values(1).isin(['', '<=', '=='])
            z = z.loc[:,indPos]
            A = X.loc[:,indPos].values
            # Scale conjunction matrix to account for non-uniform penalties
            A = A * self.lambda0 / (self.lambda0 + self.lambda1 * z.sum().values)
        if self.useOrd:
            self.namesOrd = Xstd.columns
            numOrd = Xstd.shape[1]
            # Scale ordinal features to have similar std as "average" binary feature
            Astd = 0.4 * Xstd.values
        # Iteration counter
        self.it = 0
        # Logistic regression object
        lr = LogisticRegression(
            penalty='l1',
            C=1/(n*self.lambda0),
            solver='saga',
            multi_class='ovr',
            max_iter=self.maxSolverIter)

        self.p = y.mean()
        if self.init0:
            # Initial residual
            r = (self.p - y) / n
            # Derivative w.r.t. intercept term
            UB = min(r.sum(), 0)
        else:
            # Fit logistic regression model
            if self.useOrd:
                B = np.concatenate((Astd, A), axis=1)
                lr.fit(B, y)
                # Initial residual
                r = (lr.predict_proba(B)[:,1] - y) / n
            else:
                lr.fit(A, y)
                # Initial residual
                r = (lr.predict_proba(A)[:,1] - y) / n
            # Most "negative" subderivative among current variables (undo scaling)
            UB = -np.abs(np.dot(r, A))
            UB *= (self.lambda0 + self.lambda1 * z.sum().values) / self.lambda0
            UB += self.lambda0 + self.lambda1 * z.sum().values
            UB = min(UB.min(), 0)

        # Beam search for conjunctions with subdifferentials that exclude zero
        vp, zp, Ap = beam_search_K1(r, X, self.lambda0, self.lambda1,
            UB=UB, B=self.B, wLB=self.wLB, eps=self.eps, stopEarly=self.stopEarly)
        vn, zn, An = beam_search_K1(-r, X, self.lambda0, self.lambda1,
            UB=UB, B=self.B, wLB=self.wLB, eps=self.eps, stopEarly=self.stopEarly)
        v = np.append(vp, vn)

        while (v < UB).any() and (self.it < self.iterMax):
            # Subdifferentials excluding zero exist, continue
            self.it += 1
            zNew = pd.concat([zp, zn], axis=1, ignore_index=True)
            Anew = np.concatenate((Ap, An), axis=1)

            # K conjunctions with largest subderivatives in absolute value
            idxLargest = np.argsort(v)[:self.K]
            v = v[idxLargest]
            zNew = zNew.iloc[:,idxLargest]
            Anew = Anew[:,idxLargest]
            # Scale new conjunction matrix to account for non-uniform penalties
            Anew = Anew * self.lambda0 / (self.lambda0 + self.lambda1 * zNew.sum().values)

            # Add to existing conjunctions
            z = pd.concat([z, zNew], axis=1, ignore_index=True)
            A = np.concatenate((A, Anew), axis=1)
            # Fit logistic regression model
            if self.useOrd:
                B = np.concatenate((Astd, A), axis=1)
                lr.fit(B, y)
                # Residual
                r = (lr.predict_proba(B)[:,1] - y) / n
            else:
                lr.fit(A, y)
                # Residual
                r = (lr.predict_proba(A)[:,1] - y) / n

            # Most "negative" subderivative among current variables (undo scaling)
            UB = -np.abs(np.dot(r, A))
            UB *= (self.lambda0 + self.lambda1 * z.sum().values) / self.lambda0
            UB += self.lambda0 + self.lambda1 * z.sum().values
            UB = min(UB.min(), 0)

            # Beam search for conjunctions with subdifferentials that exclude zero
            vp, zp, Ap = beam_search_K1(r, X, self.lambda0, self.lambda1,
                UB=UB, B=self.B, wLB=self.wLB, eps=self.eps, stopEarly=self.stopEarly)
            vn, zn, An = beam_search_K1(-r, X, self.lambda0, self.lambda1,
                UB=UB, B=self.B, wLB=self.wLB, eps=self.eps, stopEarly=self.stopEarly)
            v = np.append(vp, vn)

        # Restrict model to conjunctions with nonzero coefficients
        try:
            idxNonzero = np.where(np.abs(lr.coef_) > self.eps)[1]
            if self.useOrd:
                # Nonzero indices of standardized and rule features
                self.idxNonzeroOrd = idxNonzero[idxNonzero < numOrd]
                nnzOrd = len(self.idxNonzeroOrd)
                idxNonzeroRules = idxNonzero[idxNonzero >= numOrd] - numOrd
                if self.debias and len(idxNonzero):
                    # Re-fit logistic regression model with effectively no regularization
                    z = z.iloc[:,idxNonzeroRules]
                    lr.C = 1 / self.eps
                    lr.fit(B[:,idxNonzero], y)
                    idxNonzero = np.where(np.abs(lr.coef_) > self.eps)[1]
                    # Nonzero indices of standardized and rule features
                    idxNonzeroOrd2 = idxNonzero[idxNonzero < nnzOrd]
                    self.idxNonzeroOrd = self.idxNonzeroOrd[idxNonzeroOrd2]
                    idxNonzeroRules = idxNonzero[idxNonzero >= nnzOrd] - nnzOrd
                self.z = z.iloc[:,idxNonzeroRules]
                lr.coef_ = lr.coef_[:,idxNonzero]
            else:
                if self.debias and len(idxNonzero):
                    # Re-fit logistic regression model with effectively no regularization
                    z = z.iloc[:,idxNonzero]
                    lr.C = 1 / self.eps
                    lr.fit(A[:,idxNonzero], y)
                    idxNonzero = np.where(np.abs(lr.coef_) > self.eps)[1]
                self.z = z.iloc[:,idxNonzero]
                lr.coef_ = lr.coef_[:,idxNonzero]
        except AttributeError:
            # Model has no coefficients except intercept
            self.z = z
        self.lr = lr

    def compute_conjunctions(self, X):
        """Compute conjunctions of features as specified in self.z.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
        Returns:
            array: A -- Feature conjunction values, shape (X.shape[0], self.z.shape[1])
        """
        try:
            A = 1 - (((1 - X) @ self.z) > 0)
        except AttributeError:
            print("Attribute 'z' does not exist, please fit model first.")
        # Scale conjunctions as done in training
        A = A * self.lambda0 / (self.lambda0 + self.lambda1 * self.z.sum().values)
        return A

    def predict_proba(self, X, Xstd=None):
        """Predict probabilities of Y=1.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
            Xstd (DataFrame, optional): Standardized numerical features
        Returns:
            array: p -- Predicted probabilities
        """
        try:
            if self.lr.coef_.shape[1]:
                # Compute conjunctions of features
                A = self.compute_conjunctions(X)
                if self.useOrd:
                    # Selected ordinal features scaled as in training
                    Astd = 0.4 * Xstd.values[:,self.idxNonzeroOrd]
                    # Predict probabilities
                    B = np.concatenate((Astd, A), axis=1)
                    return self.lr.predict_proba(B)[:,1]
                else:
                    # Predict probabilities
                    return self.lr.predict_proba(A)[:,1]
            else:
                return np.full(X.shape[0], self.p)
        except AttributeError:
            return np.full(X.shape[0], self.p)

    def predict(self, X, Xstd=None):
        """Predict class labels.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
            Xstd (DataFrame, optional): Standardized numerical features
        Returns:
            array: yhat -- Predicted labels
        """
        try:
            if self.lr.coef_.shape[1]:
                # Compute conjunctions of features
                A = self.compute_conjunctions(X)
                if self.useOrd:
                    # Selected ordinal features scaled as in training
                    Astd = 0.4 * Xstd.values[:,self.idxNonzeroOrd]
                    # Predict probabilities
                    B = np.concatenate((Astd, A), axis=1)
                    return self.lr.predict(B)
                else:
                    # Predict labels
                    return self.lr.predict(A)
            else:
                return np.full(X.shape[0], self.p > 0.5, dtype=int)
        except AttributeError:
            return np.full(X.shape[0], self.p > 0.5, dtype=int)

    def explain(self, maxCoeffs=None, highDegOnly=False, prec=2):
        """Return DataFrame holding model features and their coefficients.
        
        Args:
            maxCoeffs (int, optional): Maximum number of rules/numerical features to show
            highDegOnly (bool, optional): Only show higher-degree rules
            prec (int, optional): Number of decimal places to show for floating-value thresholds
        Returns:
            DataFrame: dfExpl -- Rules/numerical features and their coefficients
        """
        # Number of ordinal features used
        if self.useOrd:
            nnzOrd = len(self.idxNonzeroOrd)
        else:
            nnzOrd = 0
        if highDegOnly:
            # Restrict to higher-degree rules
            coeffs = self.lr.coef_[0, nnzOrd:][self.z.sum() > 1]
            nnzOrd = 0
            z = self.z.loc[:, self.z.sum() > 1]
        else:
            coeffs = self.lr.coef_[0,:]
            z = self.z
        # Initialize DataFrame to be printed
        truncate = (maxCoeffs is not None) and (len(coeffs) > maxCoeffs)
        nRows = maxCoeffs + 1 if truncate else len(coeffs) + 1
        dfExpl = pd.DataFrame(index=range(nRows), columns=['rule','coefficient'])

        # Intercept term
        dfExpl.at[0, 'rule'] = '(intercept)'
        dfExpl.at[0, 'coefficient'] = self.lr.intercept_[0]
        # Sort coefficients from largest to smallest
        idxSort = np.abs(coeffs).argsort()[:-nRows:-1]
        dfExpl.loc[1:, 'coefficient'] = coeffs[idxSort]
        # Iterate over sorted coefficients
        for (row, i) in enumerate(idxSort):
            if i < nnzOrd:
                # Ordinal feature
                dfExpl.at[row+1, 'rule'] = self.namesOrd[self.idxNonzeroOrd[i]]
            else:
                # MultiIndex of features participating in rule i
                idxFeat = z.index[z.iloc[:,i - nnzOrd] > 0]
                # String representations of features
                strFeat = idxFeat.get_level_values(0) + ' ' + idxFeat.get_level_values(1)\
                    + ' ' + idxFeat.get_level_values(2).to_series()\
                    .apply(lambda x: ('{:.' + str(prec) + 'f}').format(x) if type(x) is float else str(x))
                # String representation of rule
                dfExpl.at[row+1, 'rule'] = strFeat.str.cat(sep=' AND ')

        # Prepare and return dataframe with model features and coefficients
        if nnzOrd:
            dfExpl.rename(columns={'rule': 'rule/numerical feature'}, inplace=True)

        return dfExpl

    def visualize(self, Xorig, fb, features=None):
        """Plot generalized additive model component, which includes first-degree rules 
        and linear functions of unbinarized ordinal features but excludes higher-degree rules.
        
        Args:
            Xorig (DataFrame): Original unbinarized features
            fb: FeatureBinarizer object used to binarize features
            features (list, optional): Subset of features to be plotted
        """
        # Number of ordinal features used
        if self.useOrd:
            nnzOrd = len(self.idxNonzeroOrd)
        else:
            nnzOrd = 0

        # Initialize terms Series and x values for plots
        terms = pd.Series(index=pd.MultiIndex.from_arrays([[],[],[]], names=self.z.index.names))
        xPlot = {}

        # Iterate over ordinal features
        for i in range(nnzOrd):
            # Restrict to specified features
            f = self.namesOrd[self.idxNonzeroOrd[i]]
            if (features is not None) and (f not in features):
                continue
            # Append term
            terms = terms.append(pd.Series(self.lr.coef_[0,i], index=[(f,'','')]))
            # Initialize x values with min and max
            xPlot[f] = [Xorig[f].min(), Xorig[f].max()]

        # Iterate over first-degree rules
        for i in range(self.z.shape[1]):
            if self.z.iloc[:,i].sum() > 1:
                continue
            # MultiIndex of rule
            idxTerm = self.z.index[self.z.iloc[:,i] > 0]
            (f, o, v) = idxTerm[0]
            # Restrict to specified features
            if (features is not None) and (f not in features):
                continue
            # Append new term
            terms = terms.append(pd.Series(self.lr.coef_[0,i+nnzOrd], index=idxTerm))
            # Update x values
            if f not in xPlot:
                if o in ['<=','>']:
                    # Ordinal feature, initialize with min and max
                    xPlot[f] = [Xorig[f].min(), Xorig[f].max()]
                else:
                    # Categorical feature, use all values
                    xPlot[f] = np.sort(Xorig[f].unique())
            if o in ['<=','>']:
                # Append values around threshold
                xPlot[f].extend([v-self.eps, v+self.eps])

        # Initialize y values for plots and variance calculation
        yPlot = {}
        yVar = pd.DataFrame(0., index=Xorig.index, columns=xPlot.keys())
        plotLine = pd.Series(False, index=xPlot.keys())
        # Iterate over GAM features
        for f in xPlot.keys():
            # Sort x values
            xPlot[f] = np.sort(np.array(xPlot[f]))
            yPlot[f] = np.zeros_like(xPlot[f], dtype=float)
            # Iterate over terms involving feature
            for ((o,v), c) in terms[f].iteritems():
                if o == '':
                    if self.useOrd and (f in fb.ordinal):
                        # Add linear function of standardized feature with same factor of 0.4
                        idxf = fb.ordinal.index(f)
                        yPlot[f] += 0.4 * c * (xPlot[f] - fb.scaler.mean_[idxf]) / fb.scaler.scale_[idxf]
                        yVar[f] += 0.4 * c * (Xorig[f] - fb.scaler.mean_[idxf]) / fb.scaler.scale_[idxf]
                        plotLine[f] = True
                    else:
                        # Binary feature, add indicator function
                        yPlot[f] += c * (xPlot[f] == fb.maps[f].index[1])
                        yVar[f] += c * (Xorig[f] == fb.maps[f].index[1])
                elif o == '<=':
                    # Add step function
                    yPlot[f] += c * (xPlot[f] <= v)
                    yVar[f] += c * (Xorig[f] <= v)
                    plotLine[f] = True
                elif o == '>':
                    # Add step function
                    yPlot[f] += c * (xPlot[f] > v)
                    yVar[f] += c * (Xorig[f] > v)
                    plotLine[f] = True
                elif o == '==':
                    # Add indicator function
                    yPlot[f] += c * (xPlot[f].astype(str) == v)
                    yVar[f] += c * (Xorig[f].astype(str) == v)
                elif o == '!=':
                    # Add indicator function
                    yPlot[f] += c * (xPlot[f].astype(str) != v)
                    yVar[f] += c * (Xorig[f].astype(str) != v)
                elif o == 'not':
                    # Binary feature, add indicator function
                    yPlot[f] += c * (xPlot[f] == fb.maps[f].index[0])
                    yVar[f] += c * (Xorig[f] == fb.maps[f].index[0])

        # Plot in order of descending variance
        figs = {}
        yVar2 = yVar.var().sort_values(ascending=False)
        for f in yVar2.index:
            figs[f] = plt.figure()
            ax = figs[f].add_subplot(111)
            if plotLine[f]:
                ax.plot(xPlot[f], yPlot[f])
            else:
                ax.bar(xPlot[f], yPlot[f])
                ax.xaxis.set_ticks(xPlot[f])
            plt.xlabel(f)
            plt.ylabel('contribution to log-odds of Y=1')

        return figs, yVar2
