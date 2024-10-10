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
    def __init__(self):
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
        instances_one_not = self.encoder.transform(instances)
        if num_counterfactuals > self.train_unaffected.shape[0]:
            warnings.warn(f"{num_counterfactuals} were requested, but only {self.train_unaffected.shape[0]} unaffected instances given. Taking all.")
            num_counterfactuals = self.train_unaffected.shape[0]
        nn = NearestNeighbors(n_neighbors=num_counterfactuals).fit(self.train_unaffected_one_hot)
        distances, indices = nn.kneighbors(instances_one_not)

        cfs = [self.train_unaffected.iloc[row] for row in indices]

        return pd.concat(cfs, ignore_index=False)


