from enum import Enum
from typing import List, Union, Callable
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from aix360.algorithms.lbbe import LocalBBExplainer
from aix360.algorithms.nncontrastive.embedding import AEEmbedding, JointAEEmbedding


class EmbeddingType(Enum):
    SUPERVISED = "class label guided embedding"
    UNSUPERVISED = "unsupervised embedding"


class NearestNeighborContrastiveExplainer(LocalBBExplainer):
    """The Nearest Neighbor (NN) Contrastive algorithm uses an auto-encoder to train low dimensional
    representation for each data point, for the nearest neighbor search. The dimensional reduction improves
    the reliability of the neighborhood search. Along with dimensionality reduction, the
    current implementation also allows for imposing class structure-driven orientation of the
    embedding space. For example, in a loan application, a high-income applicant and a
    low-income applicant may have very different evaluation criteria. The auto-encoder uses
    high-income and low-income tag classes during the auto-encoder training, ensuring
    instances with the same tags are in close neighborhoods.

    The implementation allows exemplar selection in two ways, (1) contrastive exemplar
    selection is guided by a model prediction, and (2) the user explicitly provides an
    exemplar with a different class tag than the query instance (model-free). Given a query
    instance, the resulting explanation is a set of nearest neighbor exemplars with different
    class tags than the query instances."""

    def __init__(
        self,
        model: Callable = None,
        n_classes: int = 2,
        metric: str = "euclidean",  # TODO
        neighbors: int = 3,
        embedding_type: Union[str, EmbeddingType] = EmbeddingType.UNSUPERVISED,
        embedding_dim: int = 8,
        category_enc_dim: int = 3,
        category_encoding: str = "ohe",  # supported values ohe|label
        numeric_scaling: str = None,  # supported values minmax|standard|quantile
        layers_config: List[int] = [16, 16],
        encoder_activation: str = "relu",
        decoder_activation: str = "relu",
        embedding_activation: str = "tanh",
        encoder_kernel_regularizer: str = "l1",
        encoder_kernel_initializer: str = "glorot_uniform",
        encoder_bias_initializer: str = "zeros",
        encoder_activity_regularizer: str = None,
        decoder_kernel_regularizer: str = "l1",
        decoder_kernel_initializer: str = "glorot_uniform",
        decoder_bias_initializer: str = "zeros",
        decoder_activity_regularizer: str = None,
        decoder_last_layer_activation: str = "linear",
        embedding_activity_regularizer: str = None,
        classifier_layers: List[int] = [16],
        classifier_activation: str = "relu",
        classifier_kernel_regularizer: str = None,
        classifier_kernel_initilizer: str = "glorot_uniform",
        classifier_bias_initializer: str = "zeros",
    ):
        """
        NearestNeighborContrastiveExplainer initialization.

        Args:
            model (Callable): Classification Model which will be used for contrastive
                explanation.
            n_classes (int): Number of classes the classification produces.
            metric (str): Distance metric for neighborhood finding. This metric is used
                to find neighborhood in the embedding space. The implementation internally
                uses Scipy KDTree for neighborhood search. See the documentation of
                scipy.spatial.distance and the metrics listed in distance_metrics for more
                information. Defaults to euclidean.
            neighbors (int): Number of neighbors to fetch for producing the explanation.
                The NearestNeighborContrastiveExplainer uses these neighbors to produce
                explanation. In order to understand the variety in the neighborhood profile,
                higher value is suggested for this parameter, which impacts the size of the
                explanation produced. For fast greedy explanation lower value is suggested
                for this parameter. Defaults to 3.
            embedding_type (Union[str, EmbeddingType]): This parameter controls the nature of
                the embedding produced. It can be set to supervised (EmbeddingType.SUPERVISED)
                or unsupervised (EmbeddingType.UNSUPERVISED). The unsupervised embedding ensures
                data distribution compliance, while supervised embedding allow imposing further
                structural constraints to the embedding by provided business tags during the model
                fit step. Defaults to EmbeddingType.UNSUPERVISED.
            embedding_dim (int): Dimension of the produced embedding. Lower dimension allows faster
                search, while at the cost of lossy reconstruction. An appropriate emebedding_dim
                selection depends of the data complexity and available data. Defaults to 8.
            category_enc_dim (int): Autoencoder handles categorical variable as embedding/one hot
                encoding. This parameter defines the internal dimension to be used by the auto-encoder
                to derive the categorical embedding. Defaults to 3.
            category_encoding (str): Strategy specification for categorical variable handling.
                Supported values are 'ohe' (One hot encoding) and 'label' (Uses embedding).
                Defaults to "ohe".
            numeric_scaling (str): Data scaling to be used on numeric columns for computational
                stability. This uses global scaling, i.e., applied uniformly over the entire training
                batch for all numeric columns. Supported values are minmax, standard, quantile.
                Defaults to None.
            layers_config (List[int]): This is auto-encoder internals specification. Autoencoder
                uses MLP layers to derive the embedding, this parameter specifies number of hidden
                layers in the embedding, and their respective dimensions. Defaults to [16, 16].
            encoder_activation (str): Activation function used by the auto-encoder encoding layers.
                Supports all activation function as enabled by tensorflow framework. Defaults to "relu".
            decoder_activation (str): Activation function used by the auto-encoder decoding layers.
                Support all activation functions as supported by the tensorflow framework. Defaults to
                "relu".
            embedding_activation (str): This is embedding layer activation, this can be separately
                specified than hidden layer activation. Support all tensorflow activation function.
                Defaults to "tanh".
            encoder_kernel_regularizer (str): Regularization for the encoder MLP kernel. Regularization
                results in stable prediction model. Defaults to "l1".
            encoder_kernel_initializer (str): Initialization algorithm for the MLP kernel. Defaults to
                "glorot_uniform".
            encoder_bias_initializer (str): Initialization algorithm for the MLP bias. Defaults to "zeros".
            encoder_activity_regularizer (str): Encoder activity regularizer for MLP layers. Defaults to
                None.
            decoder_kernel_regularizer (str): Kernel regularizer for the decoder MLP layer. All tensorflow
                regularizer algorithm are supported. Defaults to "l1".
            decoder_kernel_initializer (str): Decoder MLP kernel weight initializer algorithm. All
                tensorflow initializer algorithms are supported. Defaults to "glorot_uniform".
            decoder_bias_initializer (str): Decoder MLP bias initializer algorithm. All tensorflow
                initializer algorithms are supported. Defaults to "zeros".
            decoder_activity_regularizer (str): Decoder activity regularization algorithm. All tensorflow
                regularizer algorithms are supported. Defaults to None.
            decoder_last_layer_activation (str): Decoder last layer activation. This layer produces the
                input reconstruction. Supports all tensorflow supported activation function. Defaults
                to "linear".
            embedding_activity_regularizer (str): Embedding activity regularization method. Uses default
                tensorflow framework, support all activity regularizer algorithm. Defaults to None.
            classifier_layers (List[int]): Supervised auto-encoder uses classification layer for the
                structural constraint on the embedding. MLP layer for this classification task. This
                specifies the dimension of the MLP layer for this classification task. Defaults to [16].
            classifier_activation (str): Supervised auto-encoder uses classification layer for
                the structural constraint on the embedding. For supervised auto-encoder activation
                of MLP layer classification layer. Defaults to "relu".
            classifier_kernel_regularizer (str): Supervised auto-encoder uses classification layer for
                the structural constraint on the embedding. This parameter describes the kernel
                regularization for supervised auto-encoder MLP layer for classification. Defaults to None.
            classifier_kernel_initilizer (str): Supervised auto-encoder uses classification layer for the
                structural constraint on the embedding. This parameter describes the kernel initialization
                algorithm for supervised auto-encoder MLP layer for the classification. Defaults to
                "glorot_uniform".
            classifier_bias_initializer (str): Describes the MLP bias initialization algorithm, for the
                classification layer of supervised auto-encoder. Defaults to "zeros".
        """
        super(NearestNeighborContrastiveExplainer, self).__init__()
        self._config = dict(
            model=model,
            n_classes=n_classes,
            metric=metric,
            neighbors=neighbors,
            embedding_type=embedding_type.name
            if isinstance(embedding_type, EmbeddingType)
            else embedding_type,
            embedding_dim=embedding_dim,
            category_enc_dim=category_enc_dim,
            category_encoding=category_encoding,
            numeric_scaling=numeric_scaling,
            layers_config=layers_config,
            encoder_activation=encoder_activation,
            decoder_activation=decoder_activation,
            embedding_activation=embedding_activation,
            encoder_kernel_regularizer=encoder_kernel_regularizer,
            encoder_kernel_initializer=encoder_kernel_initializer,
            encoder_bias_initializer=encoder_bias_initializer,
            encoder_activity_regularizer=encoder_activity_regularizer,
            decoder_kernel_initializer=decoder_kernel_initializer,
            decoder_kernel_regularizer=decoder_kernel_regularizer,
            decoder_bias_initializer=decoder_bias_initializer,
            decoder_activity_regularizer=decoder_activity_regularizer,
            decoder_last_layer_activation=decoder_last_layer_activation,
            embedding_activity_regularizer=embedding_activity_regularizer,
            classifier_layers=classifier_layers,
            classifier_activation=classifier_activation,
            classifier_kernel_regularizer=classifier_kernel_regularizer,
            classifier_kernel_initilizer=classifier_kernel_initilizer,
            classifier_bias_initializer=classifier_bias_initializer,
        )
        self._embedding = None
        self._exemplar = None
        self._neighbor_tree = None
        self._class_based_neighbor = dict()
        self.is_fitted = False

    @property
    def embedding_type(self) -> str:
        return self._config.get("embedding_type")

    @property
    def metric(self) -> str:
        return self._config.get("metric")

    @property
    def model(self) -> Callable:
        return self._config.get("model")

    @property
    def n_classes(self) -> int:
        return self._config.get("n_classes")

    def _embedding_args(self, **kwargs) -> dict:
        self._config.update(kwargs)
        opts = {}
        common_keys = [
            "embedding_dim",
            "category_enc_dim",
            "category_encoding",
            "numeric_scaling",
            "layers_config",
            "encoder_activation",
            "decoder_activation",
            "embedding_activation",
            "encoder_kernel_regularizer",
            "encoder_kernel_initializer",
            "encoder_bias_initializer",
            "encoder_activity_regularizer",
            "decoder_kernel_regularizer",
            "decoder_kernel_initializer",
            "decoder_bias_initializer",
            "decoder_activity_regularizer",
            "decoder_last_layer_activation",
            "embedding_activity_regularizer",
        ]
        for key in common_keys:
            opts[key] = self._config.get(key)
        if self.embedding_type == EmbeddingType.SUPERVISED.name:
            extra_keys = [
                "classifier_layers",
                "classifier_activation",
                "classifier_kernel_regularizer",
                "classifier_kernel_initilizer",
                "classifier_bias_initializer",
            ]
            for key in extra_keys:
                opts[key] = self._config.get(key)
        return opts

    def fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray = None,
        features: List[str] = None,
        categorical_features: List[str] = [],
        categorical_values: dict = {},
        epochs: int = 5,
        batch_size: int = 128,
        verbose: int = 0,
        shuffle: bool = True,
        validation_fraction: float = 0,
        max_training_records: int = 10000,
        exemplars: pd.DataFrame = None,
        random_seed: int = None,
        **kwargs,
    ):
        """
        Fit the explainer.

        Args:
            x (pd.DataFrame): Training data.
            y (np.ndarray): If provided these data labels is used for training supervised
                auto-encoder. This labels need not be same as data target classes from target
                class label. Defaults to None.
            features (List[str]): Names of the features to be used. If not specified all the
                columns in the training data will be used as features. Defaults to None.
            categorical_features (List[str]): Names of the categorical features in the data.
                This must match the column names. Defaults to [].
            categorical_values (dict): Lookup dictionary for all categorical variables, list
                of possible categorical values. Defaults to {}.
            epochs (int): Number of epochs to be used for auto-encoder training. Defaults to 5.
            batch_size (int): Batch-size for the auto-encoder training. Should be smaller than
                the available data in training data. Defaults to 128.
            verbose (int): Log verbosity. Defaults to 0.
            shuffle (bool): Shuffle batch per epoch. Defaults to True.
            validation_fraction (float): Fraction specifying the validation split during encoder
                training. Defaults to 0.
            max_training_records (int): Maximum number of records to be used for the auto-encoder
                training. Enables explainer performance optimization. Defaults to 10000.
            exemplars (pd.DataFrame): Exemplar neighbors to be used to compute explanations. If
                None, training dataset will be used as exemplars. Defaults to None.
            random_seed (int): Random seed to fit auto encoder. Defaults to None

        """
        opts = self._embedding_args(**kwargs)
        cls = None
        if self.embedding_type == EmbeddingType.SUPERVISED.name:
            cls = JointAEEmbedding  # supervised
        else:
            cls = AEEmbedding  # unsupervised

        self._embedding = cls(
            data=x,
            features=features,
            categorical_features=categorical_features,
            categorical_values=categorical_values,
            max_train_records=max_training_records,
            **opts,
        )
        history = self._embedding.fit(
            x,
            y,
            epochs=epochs,
            shuffle=shuffle,
            verbose=verbose,
            batch_size=batch_size,
            validation_fraction=validation_fraction,
            random_seed=random_seed,
        )
        self.is_fitted = self._embedding.is_fitted
        if exemplars is None:
            exemplars = x
        self.set_exemplars(exemplars)
        return history

    def set_exemplars(self, x: Union[pd.DataFrame, np.ndarray]):
        """
        Set user provided exemplars to guide contrastive exploration.

        Args:
            x (Union[pd.DataFrame, np.ndarray]): Exemplar neighbors to be used to compute
                explanations.

        """
        if not self.is_fitted:
            raise RuntimeError(f"Error: exemplar can only be set post model fitting!")

        x = np.asarray(x)
        if self.model is not None:  # identify class tags for exemplars using model.
            classes = self.model(x)
            classes = np.array(classes, dtype=int).reshape(-1)
            if len(classes) != x.shape[0]:
                raise ValueError(
                    f"Error: please ensure the classification model provides integer class tag!"
                )
            for i in range(self.n_classes):
                class_idx = np.where(classes == i)[0]
                if len(class_idx) > 0:
                    self._class_based_neighbor[i] = list(class_idx)
        else:
            self._train_neighbor_tree(exemplars=x)

        if isinstance(x, np.ndarray):
            self._exemplars = pd.DataFrame(x, columns=self._embedding.features)
        else:
            self._exemplars = x.copy()

        return self

    def _train_neighbor_tree(self, exemplars):
        emb_x = self._embedding.predict(exemplars)
        self._neighbor_tree = KDTree(
            emb_x,
            leaf_size=max(1, min(25, int(0.1 * emb_x.shape[0]))),
            metric=self.metric,
        )
        return self

    def set_params(self, *argv, **kwargs):
        """Set parameters for the explainer."""
        self._config.update(kwargs)
        return self

    def get_params(self, *argv, **kwargs) -> dict:
        """Get parameters for the explainer."""
        return self._config.copy()

    def explain_instance(self, x, **kwargs):
        """Explain (local explanation) the model prediction for provided instance(s).

        Args:
            x (Union[pd.DataFrame, np.ndarray]): input instance to be explained.

        Additional Parameters:
            neighbors (int): Number of neighbors
                Overrides neighbors parameter provided in the initializer.

        Returns:
            Union(List[dict], dict): explanation object
                Dictionary or list of dictionaries with keys: features,
                categorical_features, query, neighbors, distances.

        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Error: model needs to be fitted to data prior to explanation!"
            )
        x = np.asarray(x)
        k = kwargs.get("neighbors", self._config.get("neighbors"))
        single_instance = False
        if len(x.shape) == 1:
            x = x[np.newaxis, ...]
            single_instance = True
        emb_x = self._embedding.predict(x)

        class_x = None
        filtered_exemplars = self._exemplars
        if self.model is not None:
            class_x = self.model(x)
            if isinstance(class_x, (list, np.ndarray)):
                class_x = list(np.array(class_x, dtype=int).reshape(-1))
            else:
                class_x = list(np.array([class_x], dtype=int).reshape(-1))
            if len(class_x) != emb_x.shape[0]:
                raise ValueError(f"Error: inconsistent class predictor shape!")

            # class_x[0] may not be present in self._class_based_neighbor if train data is not complete.
            filtered_exemplars = self._exemplars.drop(
                index=self._class_based_neighbor.get(class_x[0], [])
            )
            self._train_neighbor_tree(exemplars=filtered_exemplars)

        dist, idx = self._neighbor_tree.query(emb_x, k=k)

        if single_instance:
            data = {}
            data["features"] = self._embedding.features
            data["categorical_features"] = self._embedding.categorical_features()
            data["query"] = x[0].tolist()
            data["neighbors"] = []
            data["distances"] = []
            idx = idx[0].tolist()

            filtered_idx = []
            filtered_dist_idx = []
            if (class_x is not None) and (
                class_x[0] in self._class_based_neighbor
            ):  # in case of model driven, filter same class neighbors.
                for k, j in enumerate(idx):
                    if j not in self._class_based_neighbor[class_x[0]]:
                        filtered_dist_idx.append(k)
                        filtered_idx.append(j)
            else:  # in case of model free, all the returned indices are valid neighbors.
                filtered_idx = idx
                filtered_dist_idx = list(range(len(idx)))

            if len(filtered_idx) > 0:
                data["neighbors"] = filtered_exemplars.iloc[
                    filtered_idx, :
                ].values.tolist()
                data["distances"] = list(dist[0][filtered_dist_idx])
            return data
        else:
            explanations = []
            for i, idx_set in enumerate(idx):
                data = {}
                data["features"] = self._embedding.features
                data["categorical_features"] = self._embedding.categorical_features()
                data["query"] = x[i].tolist()
                data["neighbors"] = []
                data["distances"] = []

                idx_set = idx_set.tolist()

                filtered_idx = []
                filtered_dist_idx = []
                if (class_x is not None) and (
                    class_x[i] in self._class_based_neighbor
                ):  # in case of model driven, filter same class neighbors.
                    for k, j in enumerate(idx_set):
                        if j not in self._class_based_neighbor[class_x[i]]:
                            filtered_dist_idx.append(k)
                            filtered_idx.append(j)
                else:  # in case of model free, all the returned indices are valid neighbors.
                    filtered_idx = idx_set
                    filtered_dist_idx = list(range(len(idx_set)))

                if len(filtered_idx) > 0:
                    data["neighbors"] = filtered_exemplars.iloc[
                        filtered_idx, :
                    ].values.tolist()
                    data["distances"] = list(dist[i][filtered_dist_idx])
                explanations.append(data)
            return explanations
