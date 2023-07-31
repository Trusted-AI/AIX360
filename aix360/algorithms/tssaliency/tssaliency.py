import warnings
import numpy as np
from typing import List, Union, Callable
from aix360.algorithms.tslbbe import TSLocalBBExplainer
from aix360.algorithms.tssaliency.gradient import mc_gradient_compute
from aix360.algorithms.tsutils.tsframe import tsFrame, to_np_array


class TSSaliencyExplainer(TSLocalBBExplainer):
    """
    Time Series Saliency (TSSaliency) Explainer is a model agnostic saliency explainer
    for time series associate tasks. The TSSaliency supports univariate and multivariate
    use cases. It explains temporal importance of different variates on the model prediction.
    TSSaliency incorporates an integrated gradient method for saliency estimation. The
    saliency measure involves the notion of a base value. For example, the base value can be
    the constant signal with average value. The saliency measure is computed by integrating
    the model sensitivity over a trajectory from the base value to the time series signal. The
    TSSaliency explainer provides variate wise contributions to model prediction at a
    temporal resolution.

    References:
        .. [#0] `Mukund Sundararajan et al. "Axiomatic Attribution for Deep Networks"
            <https://arxiv.org/pdf/1703.01365.pdf>`_
    """

    def __init__(
        self,
        model: Callable,
        input_length: int,
        feature_names: List[str],
        base_value: List[float] = None,
        n_samples: int = 50,
        gradient_samples: int = 25,
        gradient_function: Callable = None,
        random_seed: int = 22,
    ):
        """Initializer for TSSaliencyExplainer

        Args:
            model (Callable): Callable object produces a prediction as numpy array
                for a given input as numpy array. It can be a model prediction (predict/
                predict_proba) function that results a real value like probability or regressed value.
                This function must accept numpy array of shape (input_length x len(feature_names)) as
                input and result in numpy array of shape (1, -1). Currently, TSSaliency supports sinlge output
                models only. For multi-output models, you can aggregate the output using a custom
                model_wrapper. Use model wrapper classes from aix360.algorithms.tsutils.model_wrappers.
            input_length (int): length of history window used in model training.
            feature_names (List[str]): list of feature names in the input data.
            base_value (List[float]): base value to be used in saliency computation. The
                computed gradients are with respect to this base value. If None, mean value
                is used. Defaults to None.
            n_samples (int): number of path samples to be created for each input instance
                while computing saliency metric. Defaults to 50.
            gradient_samples (int): number of timeseries samples to be generated while
                computing integreated gradient on the input data. Defaults to 25.
            gradient_function (Callable): gradient function to be used in saliency (integrated
                gradient) computation. If None, mc_gradient_compute is used. Defaults to None.
            random_seed (int): random seed to get consistent results. Refer to numpy random state.
                Defaults to 22.
        """
        super(TSSaliencyExplainer, self).__init__()
        self._model = model
        self._config = dict(
            n_samples=n_samples,
            base_value=base_value,
            gradient_samples=gradient_samples,
            input_length=input_length,
            feature_names=feature_names,
            gradient_function=gradient_function,
            random_seed=random_seed,
        )
        self._is_fitted = True

    def set_params(self, *argv, **kwargs):
        """Set parameters for the explainer."""
        self._config.update(kwargs)
        return self

    def get_params(self, *argv, **kwargs) -> dict:
        """Get parameters for the explainer."""
        return self._config.copy()

    def _affine_samples(
        self,
        alpha: float,
        x_target: np.ndarray,
        x_base: np.ndarray,
    ):
        """
        Path sampling
        """
        if x_target.shape != x_base.shape:
            raise ValueError(
                f"Error: target and base should be of "
                f"same shape {x_target.shape} != {x_base.shape}"
            )

        return alpha * x_target + (1.0 - alpha) * x_base

    def explain_instance(self, ts: tsFrame, **explain_params):
        """Explain the prediction made by the time series model at a certain point in time
        (**local explanation**).

        Args:
            ts (tsFrame): Input time series signal in ``tsFrame`` format. This can
                be generated using :py:mod:`aix360.algorithms.tsframe.tsFrame`.
                A ``tsFrame`` is a pandas ``DataFrame`` indexed by ``Timestamp`` objects
                (that is ``DatetimeIndex``). Each column corresponds to an input feature.
            explain_params: Arbitrary explainer parameters.

        Returns:
            dict: explanation object
                Dictionary with input_data, saliency, feature_names, timestamps, base_value,
                instance_prediction, base_value_prediction.
        """
        return super(TSSaliencyExplainer, self).explain_instance(
            ts=ts, ts_related=None, **explain_params
        )

    def _explain_instance(
        self,
        ts: tsFrame,
        **explain_params,
    ):
        # fix seed for consistent results
        np.random.seed(self._config.get("random_seed"))

        # retrieve explainer parameters
        input_length = self._config.get("input_length")
        feature_names = self._config.get("feature_names")
        gradient_function = self._config.get("gradient_function") or mc_gradient_compute
        gradient_samples = explain_params.get(
            "gradient_samples", self._config.get("gradient_samples")
        )
        n_samples = explain_params.get("n_samples", self._config.get("n_samples"))
        x_base = explain_params.get("base_value", self._config.get("base_value"))

        timestamps = [str(t) for t in ts.index.tolist()]
        x = to_np_array(ts)  # access in numpy array format

        if x.shape[0] != input_length:
            raise ValueError(
                "Error: expecting input length {} but found {}.".format(
                    input_length, x.shape[0]
                )
            )

        if len(x.shape) == 1:
            if len(feature_names) > 1:
                raise ValueError("Error: missing features!")
        elif x.shape[-1] != len(feature_names):
            raise ValueError(
                "Error: missing variates. Expecting input data of shape ({} x {}).".format(
                    input_length, len(feature_names)
                )
            )

        if not (isinstance(x_base, np.ndarray) | isinstance(x_base, list)):
            x_base = np.mean(x, axis=0)

        x_base = np.ones_like(x) * x_base

        # compute model prediction
        instances = np.asarray([x, x_base])
        instance_predictions = None
        try:
            instance_predictions = self._model(instances)
        except Exception as ex:
            warnings.warn(
                "Batch scoring failed with error: {}. Scoring sequentially...".format(
                    ex
                )
            )
            instance_predictions = [
                self._model(instances[i]) for i in range(instances.shape[0])
            ]
            instance_predictions = np.array(instance_predictions)

        if instance_predictions is None:
            raise Exception(
                "Model prediction could not be computed for gradient samples."
            )

        score = np.zeros_like(x)
        dt = 1 / (n_samples - 1)
        for alpha in np.linspace(0, 1, n_samples):
            xs = self._affine_samples(alpha, x, x_base)  # path sampler
            g = gradient_function(x=xs, fn=self._model, n_samples=gradient_samples)
            score += g * dt

        score = (x - x_base) * score
        normalized_score = score

        # explanation object
        explanation = {
            "input_data": x,
            "saliency": normalized_score,
            "feature_names": feature_names,
            "timestamps": timestamps,
            "base_value": x_base,
            "instance_prediction": instance_predictions[0],
            "base_value_prediction": instance_predictions[1],
        }

        return explanation
