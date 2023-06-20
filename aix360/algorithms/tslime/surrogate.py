from abc import abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression


class LinearSurrogateModel:
    """Linear Interpretable Surrogate Model Wrapper."""

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    @abstractmethod
    def get_weights(self):
        pass


class LinearRegressionSurrogate(LinearSurrogateModel):
    """Linear Interpretable Surrogate Model using LinearRegression from Scikit-Learn."""

    def __init__(self):
        super(LinearRegressionSurrogate, self).__init__(LinearRegression())

    def get_weights(self):
        return self.model.coef_


def linear_surrogate_weights(
    x_perturbations: np.ndarray,
    y_perturbations: np.ndarray,
    surrogate: LinearSurrogateModel = None,
):
    """Function to compute weights from a linear interpretable model
    using provided time series pertubations."""

    if surrogate is None:
        surrogate = LinearRegressionSurrogate()

    surrogate.fit(
        x_perturbations.reshape(x_perturbations.shape[0], -1),
        y_perturbations.reshape(y_perturbations.shape[0], -1),
    )

    # retrieve weights
    return surrogate, surrogate.get_weights()
