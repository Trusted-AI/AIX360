from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class ClusteringMethod(ABC):
    """
    Abstract base class for clustering methods.
    """

    def __init__(self):
        """
        Initialize the ClusteringMethod.
        """
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Fit the clustering model on the given data.

        Parameters:
        - data (pd.DataFrame): DataFrame of input data to fit the model.
        """
        pass

    @abstractmethod
    def predict(self, instances: pd.DataFrame) -> np.ndarray:
        """
        Predict the cluster labels for the given instances.

        Parameters:
        - instances (pd.DataFrame): DataFrame of input instances.

        Returns:
        - cluster_labels (np.ndarray): Array of cluster labels for each instance.
        """
        pass


class LocalCounterfactualMethod(ABC):
    """
    Abstract base class for local counterfactual methods.
    """

    def __init__(self):
        """
        Initialize the LocalCounterfactualMethod.
        """
        pass

    @abstractmethod
    def fit(self, **kwargs):
        """
        Fit the counterfactual method.

        Parameters:
        - **kwargs: Additional keyword arguments for fitting.
        """
        pass

    @abstractmethod
    def explain_instances(
        self, instances: pd.DataFrame, num_counterfactuals: int
    ) -> pd.DataFrame:
        """
        Find the local counterfactuals for the given instances.

        Parameters:
        - instances (pd.DataFrame): DataFrame of input instances for which counterfactuals are desired.
        - num_counterfactuals (int): Number of counterfactuals to generate for each instance.

        Returns:
        - counterfactuals (pd.DataFrame): DataFrame of counterfactual instances.
        """
        pass


class GlobalCounterfactualMethod(ABC):
    """
    Abstract base class for global counterfactual methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize the LocalCounterfactualMethod.

        Parameters:
        - **kwargs: Additional keyword arguments for init.
        """
        pass

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Fit the counterfactual method.

        Parameters:
        - **kwargs: Additional keyword arguments for fitting.
        """
        pass

    @abstractmethod
    def explain_group(self, instances: pd.DataFrame) -> pd.DataFrame:
        """
        Find the global counterfactuals for the given group of instances.

        Parameters:
        - instances (pd.DataFrame, optional): DataFrame of input instances for which global counterfactuals are desired.
        If None, explain the whole group of affected instances.

        Returns:
        - counterfactuals (pd.DataFrame): DataFrame of counterfactual instances.
        """
        pass
