import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, List, Any
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
    MinMaxScaler,
)


class BaseEmbedding(ABC):
    """Abstract class for Low dimensional embedding. The class defines
    uniform API, which is used by the NNContrastive explainer internally.
    NNContrastive explainer uses BaseEmbedding to obtain low dimensional
    vector embedding of the data, which is used for Neighborhood query.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str] = None,
        categorical_features: List[str] = [],
        categorical_values: dict = {},
        max_train_records: int = 10000,
    ):
        if features is None:
            features = list(data.columns)

        if not all([f in data for f in features]) or len(features) < 1:
            raise ValueError(f"Error: invalid feature definitions!")

        if not all([f in features for f in categorical_features]):
            raise ValueError(f"Error: invalid categorical features not in data!")

        cat_feats = dict()
        for feat in categorical_features:
            tags = np.array(categorical_values.get(feat, data[feat].unique()))
            if len(data[feat].unique()) > len(tags):
                tags = data[feat].unique()
            cat_feats[feat] = tags

        self._features = features
        self._data = data[features]
        self._categorical_features = cat_feats
        self._fitted = False
        self._max_train_records = max_train_records

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def features(self) -> List[str]:
        return self._features

    def categorical_features(
        self, return_index: bool = False
    ) -> Union[List[str], List[int]]:
        return [
            i if return_index else feat
            for i, feat in enumerate(self._features)
            if feat in self._categorical_features
        ]

    def numerical_features(
        self, return_index: bool = False
    ) -> Union[List[str], List[int]]:
        return [
            i if return_index else feat
            for i, feat in enumerate(self._features)
            if feat not in self._categorical_features
        ]

    @abstractmethod
    def fit(self, x: Union[pd.DataFrame, np.ndarray], y: Any = None):
        raise NotImplementedError(f"Error: not implemented in the base class!")

    @abstractmethod
    def predict(self, x: Union[pd.DataFrame, np.ndarray]):
        raise NotImplementedError(f"Error: not implemented in the base class!")

    @abstractmethod
    def save_model(self, filename: str):
        raise NotImplementedError(f"Error: not implemented in the base class")

    @abstractmethod
    def get_config(self):
        raise NotImplementedError(f"Error: not implemented in the base class!")


class AEEmbedding(BaseEmbedding):
    """AEEmbedding is an implementation of BaseEmbedding. AEEmbedding implements
    encoder-decoder architecture. This implementation is used by the NNContrastive
    explainer, to derive low-dimensional vector embedding in an unsupervised fashion.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str] = None,
        categorical_features: List[str] = [],
        categorical_values: dict = {},
        max_train_records: int = 10000,
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
    ):
        super(AEEmbedding, self).__init__(
            data=data,
            features=features,
            categorical_features=categorical_features,
            categorical_values=categorical_values,
            max_train_records=max_train_records,
        )

        self._encoder = None
        self._decoder = None
        self._config = dict(
            embedding_dim=embedding_dim,
            category_enc_dim=category_enc_dim,
            category_encoding=category_encoding,
            numeric_scaling=numeric_scaling,
            layers_config=layers_config,
            encoder_activation=encoder_activation,
            decoder_activation=decoder_activation,
            embedding_activation=embedding_activation,
            encoder_kernel_initializer=encoder_kernel_initializer,
            decoder_kernel_initializer=decoder_kernel_initializer,
            encoder_bias_initializer=encoder_bias_initializer,
            decoder_bias_initializer=decoder_bias_initializer,
            encoder_activity_regularizer=encoder_activity_regularizer,
            decoder_activity_regularizer=decoder_activity_regularizer,
            encoder_kernel_regularization=encoder_kernel_regularizer,
            decoder_kernel_regularization=decoder_kernel_regularizer,
            decoder_last_layer_activation=decoder_last_layer_activation,
            embedding_activity_regularizer=embedding_activity_regularizer,
        )

    def preprocess(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        fit: bool = False,
    ):
        if isinstance(x, np.ndarray):
            assert (x.shape[-1] == len(self._features)) and (
                len(x.shape) == 2
            ), f"Error: expects data of shape (#, {len(self._features)})"
            x = pd.DataFrame(x, columns=self._features)

        assert isinstance(x, pd.DataFrame)
        x = x[self._features]

        cat_enc = self._config.get("category_encoding", "ohe")
        num_enc = self._config.get("numeric_scaling", "standard")

        if (not fit) and (not self.is_fitted):
            raise RuntimeError(f"Error: incorrect fit state for preprocessing!")

        if fit is True:
            processors = []
            for i, feat in enumerate(self._features):
                if feat in self._categorical_features:
                    if cat_enc == "ohe":
                        processors.append(OneHotEncoder(handle_unknown="ignore"))
                    else:
                        processors.append(
                            OrdinalEncoder(handle_unknown="use_encoded_value")
                        )
                else:
                    if num_enc == "standard":
                        processors.append(StandardScaler())
                    elif num_enc == "minmax":
                        processors.append(MinMaxScaler())
                    elif num_enc == "quantile":
                        processors.append(QuantileTransformer())
                    else:
                        processors.append(None)
            self._config["processors"] = processors
        else:
            processors = self._config.get("processors")

        x = x.values
        xx = []
        for i, feat in enumerate(self._features):
            if fit is True:
                if processors[i] is None:
                    xi = x[:, i][..., np.newaxis]
                elif isinstance(processors[i], OneHotEncoder):
                    xi = processors[i].fit_transform(x[:, i][..., np.newaxis]).todense()
                else:
                    xi = processors[i].fit_transform(x[:, i][..., np.newaxis])
            else:
                if processors[i] is None:
                    xi = x[:, i][..., np.newaxis]
                elif isinstance(processors[i], OneHotEncoder):
                    xi = processors[i].transform(x[:, i][..., np.newaxis]).todense()
                else:
                    xi = processors[i].transform(x[:, i][..., np.newaxis])
            xx.append(xi)

        x = np.concatenate(xx, axis=1).astype("float32")
        return x

    def build(
        self,
        input_shape: int,
    ):
        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError(f"Error: can not import tensorflow library!")

        embedding_dim = self._config.get("embedding_dim")

        self._encoder = tf.keras.Sequential(name="encoder")
        for i, lyr in enumerate(self._config.get("layers_config")):
            self._encoder.add(
                tf.keras.layers.Dense(
                    int(lyr),
                    activation=self._config.get("encoder_activation"),
                    kernel_regularizer=self._config.get("encoder_kernel_regularizer"),
                    bias_initializer=self._config.get("encoder_bias_initializer"),
                    kernel_initializer=self._config.get("encoder_kernel_initializer"),
                    activity_regularizer=self._config.get(
                        "encoder_activity_regularizer"
                    ),
                    name=f"encoder_layer_{i+1}",
                )
            )
        self._encoder.add(
            tf.keras.layers.Dense(
                int(embedding_dim),
                activation=self._config.get("embedding_activation"),
                kernel_regularizer=self._config.get("encoder_kernel_regularizer"),
                bias_initializer=self._config.get("encoder_bias_initializer"),
                kernel_initializer=self._config.get("encoder_kernel_initializer"),
                activity_regularizer=self._config.get("embedding_activity_regularizer"),
                name=f"embedding_layer",
            )
        )

        self._decoder = tf.keras.Sequential(name="decoder")
        for i, lyr in enumerate(self._config.get("layers_config")[::-1]):
            self._decoder.add(
                tf.keras.layers.Dense(
                    int(lyr),
                    activation=self._config.get("decoder_activation"),
                    kernel_regularizer=self._config.get("decoder_kernel_regularizer"),
                    bias_initializer=self._config.get("decoder_bias_initializer"),
                    kernel_initializer=self._config.get("decoder_kernel_initializer"),
                    activity_regularizer=self._config.get(
                        "decoder_activity_regularizer"
                    ),
                    name=f"decoder_layer_{i+1}",
                )
            )
        self._decoder.add(
            tf.keras.layers.Dense(
                input_shape,
                activation=self._config.get("decoder_last_layer_activation"),
                kernel_regularizer=self._config.get("decoder_kernel_regularizer"),
                bias_initializer=self._config.get("decoder_bias_initializer"),
                kernel_initializer=self._config.get("decoder_kernel_initializer"),
                activity_regularizer=self._config.get("decoder_activity_regularizer"),
                name=f"decode_layer",
            )
        )

    def fit(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        y: Any = None,
        epochs: int = 5,
        batch_size: int = 512,
        shuffle: bool = True,
        validation_fraction: float = 0.1,
        verbose: int = 1,
        random_seed: int = None,
    ):

        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError(f"Error: can not import tensorflow library!")

        tf.random.set_seed(random_seed)
        x = self.preprocess(x, fit=True)
        input_shape = x.shape[-1]
        self.build(input_shape=input_shape)

        x_inp = tf.keras.layers.Input(
            input_shape,
        )
        model = tf.keras.Model(
            inputs=x_inp, outputs=[self._decoder(self._encoder(x_inp))]
        )
        model.compile(loss="mean_squared_error", optimizer="adam")

        n = x.shape[0]
        x_train = x
        validation_data = None

        if n < self._max_train_records:
            if n * validation_fraction > 10:
                n = int(n * (1.0 - validation_fraction))
                x_train, x_val = x[:n], x[n:]
                validation_data = (x_val, x_val)
            history = model.fit(
                x_train,
                x_train,
                validation_data=validation_data,
                batch_size=batch_size,
                shuffle=shuffle,
                epochs=epochs,
                verbose=verbose,
            )
        else:
            from .utils import AESampler

            sampler = AESampler(
                x_train, batch_size=batch_size, max_size=self._max_train_records
            )
            history = model.fit(sampler, epochs=epochs, verbose=verbose)

        self._fitted = True
        return history

    def predict(
        self,
        x: Union[np.ndarray, pd.DataFrame],
    ):
        if not self.is_fitted:
            raise RuntimeError(f"Error: predict can not be called prior to fitting!")
        x = self.preprocess(x, fit=False)
        return self._encoder.predict(x)

    def ae_loss(
        self,
        x: Union[np.ndarray, pd.DataFrame],
    ):
        if not self.is_fitted:
            raise RuntimeError(
                f"Error: aeloss can not be called prior to model fitting!"
            )
        x = self.preprocess(x, fit=False)
        xr = self._decoder(self._encoder(x)).numpy()
        return np.sqrt(np.mean(np.square(x - xr), axis=1))

    def get_config(self):
        return self._config.copy()

    def save_model(self, filename: str):
        pass


class JointAEEmbedding(AEEmbedding):
    """JoinAEEmbedding is a supervised auto-encoder architecture. It expects
    a class label with each input data. It jointly trains an auto-encoder and
    a classifier model. The classifier model classifies the obtained embedding
    by the provided class label.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str] = None,
        categorical_features: List[str] = [],
        categorical_values: dict = {},
        max_train_records: int = 10000,
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
        super(JointAEEmbedding, self).__init__(
            data=data,
            features=features,
            categorical_features=categorical_features,
            categorical_values=categorical_values,
            embedding_dim=embedding_dim,
            category_enc_dim=category_enc_dim,
            category_encoding=category_encoding,
            max_train_records=max_train_records,
            numeric_scaling=numeric_scaling,
            layers_config=layers_config,
            encoder_activation=encoder_activation,
            decoder_activation=decoder_activation,
            embedding_activation=embedding_activation,
            encoder_kernel_regularizer=encoder_kernel_regularizer,
            encoder_kernel_initializer=encoder_kernel_initializer,
            encoder_bias_initializer=encoder_bias_initializer,
            encoder_activity_regularizer=encoder_activity_regularizer,
            decoder_kernel_regularizer=decoder_kernel_regularizer,
            decoder_kernel_initializer=decoder_kernel_initializer,
            decoder_bias_initializer=decoder_bias_initializer,
            decoder_activity_regularizer=decoder_activity_regularizer,
            decoder_last_layer_activation=decoder_last_layer_activation,
            embedding_activity_regularizer=embedding_activity_regularizer,
        )
        self._config.update(
            dict(
                classifier_layers=classifier_layers,
                classifier_activation=classifier_activation,
                classifier_kernel_initilizer=classifier_kernel_initilizer,
                classifier_kernel_regularizer=classifier_kernel_regularizer,
                classifier_bias_initializer=classifier_bias_initializer,
            )
        )
        self._classifier = None

    def build(
        self,
        input_shape: int,
        n_classes: int,
    ):
        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError(f"Error: can not import tensorflow library!")

        super(JointAEEmbedding, self).build(input_shape=input_shape)
        self._classifier = tf.keras.Sequential(name="classifier")
        for i, lyr in enumerate(self._config.get("classifier_layers")):
            self._classifier.add(
                tf.keras.layers.Dense(
                    int(lyr),
                    activation=self._config.get("classifier_activation"),
                    kernel_regularizer=self._config.get(
                        "classifier_kernel_regularizer"
                    ),
                    kernel_initializer=self._config.get(
                        "classifier_kernel_initializer"
                    ),
                    bias_initializer=self._config.get("classifier_bias_initializer"),
                    name=f"classifier_layer_{i+1}",
                )
            )
        self._classifier.add(
            tf.keras.layers.Dense(
                int(n_classes), activation="softmax", name="class_activation"
            )
        )

    def fit(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        epochs: int = 5,
        batch_size: int = 128,
        shuffle: bool = True,
        validation_fraction: float = 0.1,
        verbose: int = 1,
        random_seed: int = None,
    ):

        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError(f"Error: can not import tensorflow library!")

        tf.random.set_seed(random_seed)
        x = self.preprocess(x, fit=True)
        y = np.array(y).astype("float32")

        assert (
            x.shape[0] == y.shape[0]
        ), f"Error: train x and y size does not match {x.shape[0]} != {y.shape[0]}"
        assert (
            len(y.shape) == 2
        ), f"Error: expects dense representation of classification tags!"

        input_shape = x.shape[-1]
        n_classes = y.shape[-1]
        self.build(input_shape=input_shape, n_classes=n_classes)

        n = x.shape[0]
        n = int(n * (1.0 - validation_fraction))
        x_train, x_val = x[:n], x[n:]
        y_train, y_val = y[:n], y[n:]
        x_inp = tf.keras.layers.Input(
            input_shape,
        )
        model = tf.keras.Model(
            inputs=x_inp,
            outputs=[
                self._decoder(self._encoder(x_inp)),
                self._classifier(self._encoder(x_inp)),
            ],
        )

        model.compile(
            loss=["mean_squared_error", "categorical_crossentropy"], optimizer="adam"
        )

        if n < self._max_train_records:
            history = model.fit(
                x_train,
                [x_train, y_train],
                validation_data=(x_val, [x_val, y_val]),
                batch_size=batch_size,
                shuffle=shuffle,
                epochs=epochs,
                verbose=verbose,
            )
        else:
            from .utils import JointAESampler

            sampler = JointAESampler(
                x_train,
                y_train,
                batch_size=batch_size,
                max_size=self._max_train_records,
            )
            history = model.fit_generator(
                sampler.train_generator, epochs=epochs, verbose=verbose
            )
        self._fitted = True
        return history
