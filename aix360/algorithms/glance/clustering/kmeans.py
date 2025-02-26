from ..base import ClusteringMethod
from sklearn.cluster import KMeans


class KMeansMethod(ClusteringMethod):
    """
    Implementation of a clustering method using KMeans.

    This class provides an interface to apply KMeans clustering to a dataset.
    """

    def __init__(self, num_clusters, random_seed):
        """
        Initializes the KMeansMethod class.

        Parameters:
        ----------
        num_clusters : int
            The number of clusters to form as well as the number of centroids to generate.
        random_seed : int
            A seed for the random number generator to ensure reproducibility.
        """

        self.num_clusters = num_clusters
        self.random_seed = random_seed
        self.model = KMeans()

    def fit(self, data):
        """
        Fits the KMeans model on the provided dataset.

        Parameters:
        ----------
        data : array-like or sparse matrix, shape (n_samples, n_features)
            Training instances to cluster.

        Returns:
        -------
        None
        """
        self.model = KMeans(
            n_clusters=self.num_clusters, n_init=10, random_state=self.random_seed
        )
        self.model.fit(data)

    def predict(self, instances):
        """
        Predicts the nearest cluster each sample in the provided data belongs to.

        Parameters:
        ----------
        instances : array-like or sparse matrix, shape (n_samples, n_features)
            New data to predict.

        Returns:
        -------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.model.predict(instances)
