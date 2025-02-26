from typing import List
import warnings

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from ..base import LocalCounterfactualMethod
from ..utils.action import extract_actions_pandas, apply_actions_pandas_rows

class NearestNeighborMethod(LocalCounterfactualMethod):
    """
    NearestNeighborMethod is a local counterfactual method that finds the nearest unaffected neighbors in the training dataset to explain instances by generating counterfactuals.

    This method identifies instances in the training set where the model prediction remains unaffected, 
    and uses the nearest neighbors (based on feature similarity) to generate counterfactual explanations for new instances.

    Methods:
    --------
    __init__():
        Initializes the NearestNeighborMethod instance.

    fit(model, data, outcome_name, continuous_features, feat_to_vary, random_seed=13):
        Fits the method to the training data by identifying unaffected instances based on model predictions and preparing the feature encoding for nearest neighbor searches.

    explain_instances(instances, num_counterfactuals):
        Finds and returns the nearest unaffected neighbors for each instance, generating the specified number of counterfactual explanations.
    """
    def __init__(self):
        """
        Initializes a new instance of the NearestNeighborMethod class.
        """
        super().__init__()

    def fit(
        self,
        model,
        data: pd.DataFrame,
        outcome_name: str,
        continuous_features: List[str],
        feat_to_vary: List[str],
        random_seed=13,
    ):
        """
        Fits the NearestNeighborMethod by identifying unaffected instances in the training dataset and preparing feature encodings for counterfactual search.

        Parameters:
        ----------
        model : object
            A machine learning model with a `predict` method that outputs binary predictions (0 or 1).
        data : pd.DataFrame
            A dataset containing the features and outcome variable used for fitting the method.
        outcome_name : str
            The name of the outcome column in the dataset.
        continuous_features : List[str]
            A list of continuous (numerical) feature column names.
        feat_to_vary : List[str]
            A list of features allowed to vary when generating counterfactuals.
        random_seed : int, optional
            Seed for random number generation to ensure reproducibility, by default 13.
        """
        X, y = data.drop(columns=[outcome_name]), data[outcome_name]
        self.numerical_features = continuous_features
        self.categorical_features = X.columns.difference(continuous_features).tolist()

        self.encoder = ColumnTransformer(
            [("ohe", OneHotEncoder(sparse_output=False), self.categorical_features)],
            remainder="passthrough",
        ).fit(X)

        train_preds = model.predict(X)
        self.train_unaffected = X[train_preds == 1]
        self.train_unaffected_one_hot = self.encoder.transform(self.train_unaffected)
        
        self.random_seed = random_seed
        self.feat_to_vary = feat_to_vary

    def explain_instances(
        self, instances: pd.DataFrame, num_counterfactuals: int
    ) -> pd.DataFrame:
        """
        Generates counterfactual explanations for the provided instances by finding the nearest unaffected neighbors in the training data.

        Parameters:
        ----------
        instances : pd.DataFrame
            DataFrame containing the instances for which counterfactual explanations are needed.
        num_counterfactuals : int
            The number of counterfactuals to generate for each instance.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the nearest unaffected neighbors (counterfactuals) for each instance.
        
        Notes:
        ------
        - If the requested number of counterfactuals exceeds the number of available unaffected instances, a warning is raised, and all unaffected instances are used.
        - Nearest neighbors are determined using a one-hot encoded feature representation.
        """
        instances_one_not = self.encoder.transform(instances)
        if num_counterfactuals > self.train_unaffected.shape[0]:
            warnings.warn(f"{num_counterfactuals} were requested, but only {self.train_unaffected.shape[0]} unaffected instances given. Taking all.")
            num_counterfactuals = self.train_unaffected.shape[0]
        nn = NearestNeighbors(n_neighbors=num_counterfactuals).fit(self.train_unaffected_one_hot)
        distances, indices = nn.kneighbors(instances_one_not)

        cfs = [self.train_unaffected.iloc[row] for row in indices]

        return pd.concat(cfs, ignore_index=False)


