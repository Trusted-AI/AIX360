from typing import Callable
import numpy as np


class Model:
    def __init__(self, model: Callable):
        self.model = model

    def predict(self, X: np.ndarray, **kwargs):
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X: np.ndarray, **kwargs):
        return self.model.predict_proba(X, **kwargs)


class Anomaly_Detection_Model(Model):
    def __init__(self, model: Callable, scoring_function: str):
        super().__init__(model)
        self.scoring_function = scoring_function

    def predict(self, X: np.ndarray, **kwargs):
        return getattr(self.model, self.scoring_function)(X, **kwargs)


class Classification_Model(Model):
    def __init__(
        self,
        model: Callable,
        class_pos: int = 0,
    ):
        super().__init__(model)
        self.class_pos = int(class_pos)

    def predict(self, X: np.ndarray, **kwargs):
        predictions = self.model.predict(X, **kwargs)
        predictions = np.argmax(predictions, axis=1).reshape(-1, 1)
        return predictions

    def predict_proba(self, X: np.ndarray, **kwargs):
        predictions = self.model.predict_proba(X, **kwargs)[:, self.class_pos]
        predictions = predictions.reshape(-1, 1)
        return predictions


class Forecaster(Model):
    def __init__(
        self,
        model: Callable,
        forecast_function: str = "forecast",
        reduce_function: Callable = None,
    ):
        super().__init__(model)
        self.forecast_function = forecast_function
        self.reduce_function = reduce_function
        if self.reduce_function is None:
            self.reduce_function = lambda X: np.mean(X, axis=0)

    def predict(self, X: np.ndarray, **kwargs):
        forecast = getattr(self.model, self.forecast_function)(
            X, **kwargs
        )  # of length forecast horizon
        forecast = np.asarray(forecast)
        if (forecast.shape[0] > 1) and (
            forecast.shape[0] == X.shape[0]
        ):  # batch forecast
            return forecast
        else:
            forecast = forecast.reshape(-1, 1)
            return np.mean(forecast, axis=0)


class Tensor_Based_Classification_Model(Classification_Model):
    def __init__(
        self,
        model: Callable,
        class_pos: int = 0,
        input_length: int = 2,
        n_features: int = 1,
    ):
        super(Tensor_Based_Classification_Model, self).__init__(
            model=model, class_pos=class_pos
        )
        self.input_length = input_length
        self.n_features = n_features

    def predict(self, X: np.ndarray, **kwargs):
        X = X.reshape(-1, self.input_length, self.n_features)
        return super(Tensor_Based_Classification_Model, self).predict(X, **kwargs)

    def predict_proba(self, X: np.ndarray, **kwargs):
        X = X.reshape(-1, self.input_length, self.n_features)
        predictions = self.model.predict(X, **kwargs)[:, self.class_pos]
        predictions = predictions.reshape(-1, 1)
        return predictions


class Tensor_Based_Forecaster(Forecaster):
    def __init__(self, n_features: int, input_length: int, **kwargs):
        super(Tensor_Based_Forecaster, self).__init__(**kwargs)
        self.n_features = n_features
        self.input_length = input_length

    def predict(self, X: np.ndarray, **kwargs):
        X = X.reshape(-1, self.input_length, self.n_features)
        return super(Tensor_Based_Forecaster, self).predict(X, **kwargs)
