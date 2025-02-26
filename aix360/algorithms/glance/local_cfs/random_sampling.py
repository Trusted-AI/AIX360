import pandas as pd
from ..base import LocalCounterfactualMethod
import numpy as np
from sklearn.inspection import permutation_importance

class RandomSampling(LocalCounterfactualMethod):
    """
    RandomSampling is a local counterfactual method that generates counterfactual instances 
    through random sampling based on the distribution of features in the unaffected training data.

    This method identifies the most important features and the most frequent categories within the 
    unaffected training data to generate counterfactuals by sampling from these distributions.

    Methods:
    --------
    __init__(model, n_most_important, n_categorical_most_frequent, numerical_features, categorical_features, random_state=None):
        Initializes the RandomSampling instance with the specified parameters.

    fit(X, y):
        Fits the RandomSampling method to the provided training data by calculating feature importances and identifying unaffected instances.

    _sample_instances(n_samples, fixed_feature_values, random_state=None):
        Samples instances based on the specified feature distributions, fixing certain feature values while sampling others.

    explain(instance, num_counterfactuals, n_samples=1000, random_state=None):
        Generates counterfactual explanations for a given instance by sampling and modifying feature values.

    explain_instances(instances, num_counterfactuals, n_samples=1000, random_state=None):
        Generates counterfactuals for multiple instances by calling the explain method for each instance.
    """
    def __init__(self, model, n_most_important, n_categorical_most_frequent, numerical_features, categorical_features, random_state=None):
        """
        Initializes a new instance of the RandomSampling class.

        Parameters:
        ----------
        model : object
            A machine learning model used for predictions and feature importance evaluation.
        n_most_important : int
            The number of most important features to consider when generating counterfactuals.
        n_categorical_most_frequent : int
            The number of most frequent categories to consider for categorical features.
        numerical_features : List[str]
            A list of continuous (numerical) feature names.
        categorical_features : List[str]
            A list of categorical feature names.
        random_state : int, optional
            Seed for random number generation to ensure reproducibility, by default None.
        """
        self.model = model
        self.n_most_important = n_most_important
        self.n_categorical_most_frequent = n_categorical_most_frequent
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the RandomSampling method to the provided training data by calculating feature importances and identifying unaffected instances.

        Parameters:
        ----------
        X : pd.DataFrame
            The training dataset containing feature columns.
        y : pd.Series
            The target variable corresponding to the training dataset.

        Returns:
        -------
        self : RandomSampling
            Returns the fitted instance of RandomSampling.
        """
        self.X_ = X
        self.feature_names_ = X.columns.tolist()
        # Permutation feature importance
        result = permutation_importance(self.model, X, y, random_state=self.random_state)
        self.feature_importances_ = result.importances_mean
        top_k_indices = np.argsort(self.feature_importances_)[::-1][:self.n_most_important]
        self.top_k_features_ = X.columns[top_k_indices]

        train_preds = self.model.predict(X)
        unaffected = X[train_preds == 1]

        # Store min and max values for numerical features
        self.numeric_min_ = unaffected[self.numerical_features].min()
        self.numeric_max_ = unaffected[self.numerical_features].max()
        for f in self.numerical_features:
            if np.isnan(self.numeric_min_[f]):
                self.numeric_min_[f] = X[f].min()
            if np.isnan(self.numeric_max_[f]):
                self.numeric_max_[f] = X[f].max()

        # Get the top m most frequent categories for categorical features
        self.categorical_top_m_ = {}
        for col in self.categorical_features:
            top_categories = unaffected[col].value_counts().index[:self.n_categorical_most_frequent]
            if top_categories.empty:
                top_categories = X[col].value_counts().index[:self.n_categorical_most_frequent]
            self.categorical_top_m_[col] = top_categories

        return self

    def _sample_instances(self, n_samples: int, fixed_feature_values, random_state=None):
        """
        Samples instances based on the specified feature distributions, fixing certain feature values while sampling others.

        Parameters:
        ----------
        n_samples : int
            The number of instances to sample.
        fixed_feature_values : dict
            A dictionary of feature names and their fixed values during sampling.
        random_state : int, optional
            Seed for random number generation, by default None.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the sampled instances with the same feature structure as the original data.
        """
        if random_state is not None:
            np.random.seed(random_state)
        samples_columns = []
        for col in self.X_.columns:
            if col in fixed_feature_values:
                column = [fixed_feature_values[col]] * n_samples
            elif col in self.numerical_features:
                column = np.random.uniform(self.numeric_min_[col], self.numeric_max_[col], n_samples)
            else:
                column = np.random.choice(self.categorical_top_m_[col], n_samples)
            samples_columns.append(column)
        return pd.DataFrame({col_name: column for col_name, column in zip(self.X_.columns, samples_columns)})
    
    def explain(self, instance, num_counterfactuals, n_samples=1000, random_state=None):
        """
        Generates counterfactual explanations for a given instance by sampling and modifying feature values.

        Parameters:
        ----------
        instance : pd.DataFrame
            A single row DataFrame representing the instance for which counterfactuals are generated.
        num_counterfactuals : int
            The number of counterfactuals to generate.
        n_samples : int, optional
            The number of samples to draw for generating counterfactuals, by default 1000.
        random_state : int, optional
            Seed for random number generation, by default None.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the generated counterfactuals for the provided instance.
        
        Raises:
        -------
        ValueError
            If the input instance is not a single-row DataFrame or if its columns do not match the training dataset's columns.
        """
        # Check if instance is a single row DataFrame
        if not isinstance(instance, pd.DataFrame) or instance.shape[0] != 1:
            raise ValueError("Input must be a single row DataFrame.")

        # Check if the DataFrame columns match the features provided during initialization
        if set(instance.columns) != set(self.X_.columns):
            raise ValueError("Columns of the input instance do not match the columns used during fitting.")

        fixed_feature_values = {}
        for col in self.feature_names_:
            if col not in self.top_k_features_:
                fixed_feature_values[col] = instance[col].item()
        random_instances = self._sample_instances(n_samples, fixed_feature_values, random_state)

        # Generate copies of the query instance that will be changed one feature
        # at a time to encourage sparsity.
        cfs_df = None
        candidate_cfs = instance.apply(lambda col: col.repeat(n_samples)).reset_index(drop=True)
        # Loop to change one feature at a time, then two features, and so on.
        for num_features_to_vary in range(1, len(self.top_k_features_)+1):
            selected_features = np.random.choice(self.top_k_features_, (n_samples, 1), replace=True)
            for k in range(n_samples):
                candidate_cfs.at[k, selected_features[k][0]] = random_instances.at[k, selected_features[k][0]]
            preds = self.model.predict(candidate_cfs)
            if sum(preds) > 0:
                rows_to_add = candidate_cfs[preds == 1]

                if cfs_df is None:
                    cfs_df = rows_to_add.copy()
                else:
                    cfs_df = pd.concat([cfs_df, rows_to_add])
                cfs_df.drop_duplicates(inplace=True)
                # Always change at least 2 features before stopping
                if num_features_to_vary >= 2 and len(cfs_df) >= num_counterfactuals:
                    break

        if cfs_df is None:
            return None
        
        assert isinstance(cfs_df, pd.DataFrame)
        if len(cfs_df) > num_counterfactuals:
            cfs_df = cfs_df.sample(num_counterfactuals)
        cfs_df.reset_index(inplace=True, drop=True)
        return cfs_df

    def explain_instances(
        self, instances: pd.DataFrame, num_counterfactuals: int, n_samples=1000, random_state=None
    ) -> pd.DataFrame:
        """
        Generates counterfactuals for multiple instances by calling the explain method for each instance.

        Parameters:
        ----------
        instances : pd.DataFrame
            DataFrame containing instances for which counterfactual explanations are needed.
        num_counterfactuals : int
            The number of counterfactuals to generate for each instance.
        n_samples : int, optional
            The number of samples to draw for generating counterfactuals, by default 1000.
        random_state : int, optional
            Seed for random number generation, by default None.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the generated counterfactuals for all provided instances.
        """
        cfs = []
        for i in range(instances.shape[0]):
            cfs_instance = self.explain(instances.iloc[i:i+1], num_counterfactuals=num_counterfactuals, n_samples=n_samples, random_state=random_state)
            if cfs_instance is not None:
                cfs.append(cfs_instance)
        
        ret = pd.concat(cfs, ignore_index=False) if cfs != [] else pd.DataFrame(columns=instances.columns).astype(instances.dtypes)
        return ret
