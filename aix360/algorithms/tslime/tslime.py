import warnings
import numpy as np
import pandas as pd
from typing import Union, List, Callable
from aix360.algorithms.tslbbe import TSLocalBBExplainer
from aix360.algorithms.tsutils.tsframe import tsFrame, to_np_array
from aix360.algorithms.tslime.surrogate import (
    linear_surrogate_weights,
    LinearSurrogateModel,
)
from aix360.algorithms.tsutils.tsperturbers.perturbed_data_generator import (
    PerturbedDataGenerator,
)
from aix360.algorithms.tsutils.tsperturbers.tsperturber import (
    TSPerturber,
    BlockSelector,
)


class TSLimeExplainer(TSLocalBBExplainer):
    """Time Series Local Interpretable Model-agnostic Explainer (TSLime) is a model-agnostic local time series
    explainer. LIME (Locally interpretable Model agnostic explainer) is a popular algorithm for local
    explanation. LIME explains the model behavior by approximating the model response with linear models.
    LIME algorithm specifically assumes tabular data format, where each row is a data point, and columns
    are features. A generalization of LIME algorithm for image data uses super pixel based perturbation.
    TSLIME generalizes LIME algorithm for time series context.

    TSLIME uses time series perturbation methods to produce a local input perturbation, and linear model
    surrogate which best approximates the model response. TSLime produces an interpretable explanation.
    The explanation weights produced by the TSLime explanation indicates model local sensitivity.

    References:
        .. [#0] `Ribeiro et al. '"Why Should I Trust You?": Explaining the Predictions of Any Classifier'
            <https://arxiv.org/abs/1602.04938>`_

    """

    def __init__(
        self,
        model: Callable,
        input_length: int,
        n_perturbations: int = 2000,
        relevant_history: int = None,
        perturbers: List[Union[TSPerturber, dict]] = None,
        local_interpretable_model: LinearSurrogateModel = None,
        random_seed: int = None,
    ):
        """Initializer for TSLimeExplainer

        Args:
            model (Callable): Callable object produces a prediction as numpy array
                for a given input as numpy array. It can be a model prediction (predict/
                predict_proba) function that results a real value like probability or regressed value.
                This function must accept numpy array of shape (input_length x len(feature_names)) as
                input and result in numpy array of shape (1, -1). Currently, TSLime supports sinlge output
                models only. For multi-output models, you can aggregate the output using a custom
                model_wrapper. Use model wrapper classes from aix360.algorithms.tsutils.model_wrappers.
            input_length (int): Input (history) length used for input model.
            n_perturbations (int): Number of perturbed instance for TSExplanation. Defaults to 25.
            relevant_history (int): Interested window size for explanations. The explanation is
                computed for selected latest window of length `relevant_history`. If `input_length=20`
                and `relevant_history=10`, explanation is computed for last 10 time points. If None,
                relevant_history is set to input_length. Defaults to None.
            perturbers (List[TSPerturber, dict]): data perturbation algorithm specification by TSPerturber
                instance or dict. Allowed values for "type" key in dictionary are block-bootstrap, frequency,
                moving-average, shift. Block-bootstrap split the time series into contiguous
                chunks called blocks, for each block noise is estimated and noise is exchanged
                and added to the signal between randomly selected blocks. Moving-average perturbation
                maintains the moving mean of the time series data with the specified window length,
                but add perturbed noise with similar distribution as the data. Frequency
                perturber performs FFT on the noise, and removes random high frequency
                components from the noise estimates. Number of frequencies to be removed
                is specified by the truncate_frequencies argument. Shift perturber adds
                random upward or downward shift in the data value over time continuous
                blocks. If not provided default perturber is block-bootstrap. Defaults to None.
            local_interpretable_model (LinearSurrogateModel): Local interpretable model, a surrogate that
                is to be trained on the given input time series neighborhood. This model is used to provide
                local weights for each time point in the selected timeseries. If None, sklearn's Linear Regression
                model, aix360.algorithms.tslime.surrogate.LinearRegressionSurrogate is used. Defaults to None.
            random_seed (int): random seed to get consistent results. Refer to numpy random state.
                Defaults to None.
        """
        self.model = model

        if perturbers is None:
            perturbers = [
                dict(type="block-bootstrap"),
            ]

        block_selector = BlockSelector(start=-input_length, end=None)
        perturber = PerturbedDataGenerator(
            perturber_engines=perturbers,
            block_selector=block_selector,
        )
        self._parameters = dict()

        # Input Specification
        self.input_length = input_length

        # Surrogate training params
        self.local_interpretable_model = local_interpretable_model
        self.n_perturbations = n_perturbations
        self.perturber = perturber

        # Explanation params
        if relevant_history is None:
            relevant_history = input_length

        self.relevant_history = relevant_history
        self.random_seed = random_seed

    def get_params(self):
        return self._parameters.copy()

    def set_params(self, *argv, **kwargs):
        self._parameters.update(kwargs)
        return self

    def _ts_perturb(self, x):
        # create perturbations
        x_perturbations = None
        y_perturbations = None

        x_perturbations, _ = self.perturber.fit_transform(
            x, None, n=self.n_perturbations
        )

        x_perturbations = np.asarray(x_perturbations).astype("float")
        return x_perturbations

    def _batch_predict(self, x_perturbations):
        f_predict_samples = None

        try:
            f_predict_samples = self.model(x_perturbations)
        except Exception as ex:
            warnings.warn(
                "Batch scoring failed with error: {}. Scoring sequentially...".format(
                    ex
                )
            )
            f_predict_samples = [
                self.model(x_perturbations[i]) for i in range(x_perturbations.shape[0])
            ]
            f_predict_samples = np.array(f_predict_samples)

        return f_predict_samples

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
                Dictionary with keys: input_data, history_weights, model_prediction,
                surrogate_prediction, x_perturbations, y_perturbations.
        """
        return super(TSLimeExplainer, self).explain_instance(
            ts=ts, ts_related=None, **explain_params
        )

    def _explain_instance(
        self,
        ts: tsFrame,
        **explain_params,
    ):
        # for consistent results. Is it possible here?
        np.random.seed(self.random_seed)

        ### input validation
        if ts.shape[0] < self.input_length:
            raise ValueError(
                "Error: expecting input length {} but found {}.".format(
                    self.input_length, ts.shape[0]
                )
            )
        xc = ts[-self.input_length :]
        xc = to_np_array(xc)

        ### generate time series perturbations
        x_perturbations = self._ts_perturb(x=xc)

        ### generate y
        y_perturbations = self._batch_predict(x_perturbations)
        if y_perturbations is None:
            raise Exception(
                "Model prediction could not be computed for gradient samples."
            )

        y_perturbations = np.asarray(y_perturbations).astype("float")

        ### select k time points - relevant_history
        x_perturbations = x_perturbations[
            :, -self.relevant_history :
        ]  # consider only k time points

        xc_relevant = xc[-self.relevant_history :, :].reshape(1, -1)

        ### compute weights using a linear model
        surrogate, history_weights = linear_surrogate_weights(
            surrogate=self.local_interpretable_model,
            x_perturbations=x_perturbations,
            y_perturbations=y_perturbations,
        )

        model_prediction = self._batch_predict(xc)

        surrogate_prediction = surrogate.predict(xc_relevant)
        explanation = {
            "input_data": ts,
            "model_prediction": model_prediction,
            "surrogate_prediction": surrogate_prediction,
            "history_weights": history_weights.reshape(self.relevant_history, -1),
            "x_perturbations": x_perturbations,
            "y_perturbations": y_perturbations,
        }
        return explanation
