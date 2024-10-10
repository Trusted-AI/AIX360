from ..base import ClusteringMethod
from sklearn.cluster import KMeans


class KMeansMethod(ClusteringMethod):
    """
    Example implementation of a clustering method using KMeans.
    """

    def __init__(self, num_clusters, random_seed):
        self.num_clusters = num_clusters
        self.random_seed = random_seed
        self.model = KMeans()

    def fit(self, data):
        self.model = KMeans(
            n_clusters=self.num_clusters, n_init=10, random_state=self.random_seed
        )
        self.model.fit(data)

    def predict(self, instances):
        return self.model.predict(instances)
