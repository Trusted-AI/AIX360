import warnings
from typing import List, Union, Callable
import numpy as np
import pandas as pd
import shap
from aix360.algorithms.lbbe import LocalBBExplainer


class GroupedCEExplainer(LocalBBExplainer):
    """Grouped Conditional Expectation plots are generated for a given instance and set of features.
    They show how the model prediction is affected when a pair of features of a given instance are
    perturbed. The perturbed features can be either a provided subset of the input covariates, or
    the top K features based on the importance rank obtained by a global explainer (e.g., SHAP).
    If the user provides a single feature then the algorithm produces a standard ICE plot where the
    selected feature varies according to a linespace grid. If the user chooses more than one feature,
    then the explainer produces 3D ICE plots. Here, two features vary simultaneously according to a
    meshgrid, the output of the model is stored for each pair of values.
    """

    def __init__(
        self,
        model: Callable,
        data: Union[int, List[List[object]]],
        feature_names: List[str] = None,
        n_samples: int = 25,
        features_selected: list = None,
        top_k_features: int = -1,
        feature_importance_method: str = "SHAP",
        max_dataset_size: int = 10,
        random_seed: int = None,
        **kwargs,
    ):
        """GroupedCEExplainer initialization.

        Args:
            model (Callable): model prediction (predict/predict_proba) function that
                results a real value like probability or regressed value.
            data (List[object]): Input dataset used for model training. Feature range is computed from
                this input dataset. The dataset is used in selected feature importance methods such as
                SHAP to determine top K features for group explanation.
            feature_names (List[str]): List of valid numerical feature names in the input dataset. Defaults
                to None.
            n_samples (int, optional): Number of discrete points sampled per feature. Defaults to 25.
            features_selected (List[str], optional): List of features that will be considered in the
                explanation. If list contains single feature, GroupedCEExplainer return standard ICE
                explanation. Otherwise, returns grouped explanation for 3D ICE plots.
            top_k_features (int, optional): Top K importance features to consider if features_selected
                is an empty list. If top_k_features <= 0, all the features are selected for explanation.
                Defaults to -1.
            feature_importance_method (str,optional): Importance feature method to be used if
                top_k_features is > 0 and features_selected is empty. Defaults to 'SHAP'.
            max_dataset_size (int): maximum dataset size used during selected feature importance method
                (feature_importance_method). Defaults to 10.
        """
        self.model = model
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.max_allowed_feature = 4
        self.feature_range = {}
        self.max_dataset_size = max_dataset_size
        self.random_seed = random_seed
        super(GroupedCEExplainer, self).__init__(**kwargs)

        ## FEATURES SELECTED: Options : All features, features specified by user , use top K important features
        if features_selected is None:
            features_selected = []
        self.features_selected = features_selected
        self.top_k_features = top_k_features
        self.feature_importance_method = feature_importance_method
        self._is_fitted = False
        if self.feature_names is None:
            data = pd.DataFrame(data)
            self.feature_names = list(data.columns)
        else:
            data = pd.DataFrame(data, columns=self.feature_names)

        self.fit(data)

    def fit(self, X: pd.DataFrame, *args, **kwargs):

        np.random.seed(self.random_seed)
        # compute feature ranges from input data
        for f in self.feature_names:
            self.feature_range[f] = {
                "min": np.min(X[f].values),
                "max": np.max(X[f].values),
            }

        ## If we want to consider top K importance features
        if len(self.features_selected) == 0:
            if self.top_k_features <= 0:  ## All features
                self.features_selected = self.feature_names
            else:
                ## feature selection based on shap importance
                if self.feature_importance_method == "SHAP":  ## SHAP OPTION
                    shap_data = X
                    if shap_data.shape[0] > self.max_dataset_size:
                        shap_data = self._partition_sample(
                            np.asarray(shap_data), self.max_dataset_size
                        )
                        shap_data = pd.DataFrame(shap_data, columns=X.columns)
                    shap_exp = shap.KernelExplainer(
                        self.model,
                        shap_data,
                        link="identity",
                        algorithm="auto",
                    )
                    shap_values = shap_exp.shap_values(shap_data, nsamples=50)

                    features_importance = np.abs(shap_values).mean(axis=0).tolist()
                    features_names = self.feature_names

                    ## Save names of top K importance features
                    self.features_selected = [
                        features_names[int(_)]
                        for _ in np.argsort(np.array(features_importance))[
                            -self.top_k_features :
                        ]
                    ]
                    self.features_selected = list(set(self.features_selected))

                    print(
                        "Considering Top {} features according to SHAP: {}".format(
                            self.top_k_features, self.features_selected
                        )
                    )

        self._is_fitted = True
        return self

    def _is_numeric_feature(
        self,
        feature_name,
    ) -> bool:
        if not self._is_fitted:
            raise RuntimeError(
                f"Error: can not verify feature [{feature_name}] dtype without fit operation."
            )
        return (feature_name in self.feature_range) and (
            "max" in self.feature_range[feature_name]
        )

    def _partition_sample(
        self,
        x: np.ndarray,
        n: int,
        d: int = 0,
    ):
        """
        First recursive binary partition using feature values sampling
        of the high dimensional data.
        """
        if 2**d > n:
            i = np.random.choice(x.shape[0], 1)
            return x[i]

        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)

        dim = np.argmax(x_max - x_min)
        y = np.quantile(x[:, dim], q=0.5)
        x_lo = x[np.where(x[:, dim] <= y)[0]]
        x_hi = x[np.where(x[:, dim] > y)[0]]

        s1, s2 = None, None
        if x_lo.shape[0] > 1:
            s1 = self._partition_sample(x_lo, n, d=d + 1)
            if len(s1.shape) == 1:
                s1 = s1[np.newaxis, ...]

        if x_hi.shape[0] > 1:
            s2 = self._partition_sample(x_hi, n, d=d + 1)
            if len(s2.shape) == 1:
                s2 = s2[np.newaxis, ...]

        if (s1 is not None) and (s2 is not None):
            s = np.concatenate([s1, s2], axis=0)
        else:
            s = s1 if s1 is not None else s2
        return s[:n] if d == 0 else s

    def set_params(self, *argv, **kwargs):
        """Set parameters for the explainer."""
        self._params.update(kwargs)
        return self

    def get_params(self, *argv, **kwargs) -> dict:
        """Get parameters for the explainer."""
        return self._params.copy()

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

        if f_predict_samples is None:
            raise Exception("Model prediction could not be computed.")
        return f_predict_samples

    def explain_instance(
        self,
        instance: Union[pd.DataFrame, np.ndarray],
        **kwargs,  # CAN STAY
    ):
        """Produces local explanation of the target model for selected feature(s).

        Args:
            instance (Union[pd.DataFrame, np.ndarray]): input instance to be explained.

        Returns:
            dict: explanation object
                Dictionary with feature_name, feature_value, ice_value, current_value
                for ICE explanation. Dictionary with gce_values, x_grid, y_pred,
                current_values for GCE explanation.

        """

        np.random.seed(self.random_seed)
        return_instances = kwargs.get("return_instances", True)
        feat_perturbation = kwargs.get("feature_perturbations", {})

        if len(self.features_selected) < 1:
            raise ValueError(
                f"Error: explain_instance API expects at least one "
                f"numeric feature specification."
            )

        for feat in self.features_selected:
            if not self._is_numeric_feature(feat):
                raise ValueError(
                    f"Error: supports numeric features only, "
                    f"{feat} is not a numeric feature!"
                )

        if instance.shape[0] != 1:
            raise RuntimeError(
                f"Error: explain_instance must be invoked for single instance only"
            )

        if isinstance(instance, np.ndarray):
            instance = pd.DataFrame(instance, columns=self.feature_names)

        if not all([feat in instance for feat in self.feature_names]):
            raise ValueError(
                f"Error: expects instance indices or dataframe "
                f"with all feature columns"
            )

        instance = instance[self.feature_names]
        feature_sample = {}

        # generate perturbed feature samples
        for feat in self.features_selected:
            mn = self.feature_range[feat]["min"]
            mx = self.feature_range[feat]["max"]
            if feat in feat_perturbation:
                perturb_mn = feat_perturbation[feat]["min"]
                perturb_mx = feat_perturbation[feat]["max"]

                feature_sample[feat] = np.linspace(
                    perturb_mn, perturb_mx, self.n_samples
                )
            else:
                feature_sample[feat] = np.linspace(mn, mx, self.n_samples)

        # compute conditional expectation
        if len(self.features_selected) == 1:  # ICE
            feature_name = self.features_selected[0]
            curr_value = instance[feature_name].values[0]
            ordered_values = feature_sample[feature_name]

            x_perturbed = pd.concat([instance for _ in ordered_values]).reset_index(
                drop=True
            )
            x_perturbed.loc[:, feature_name] = ordered_values

            predictions = self._batch_predict(x_perturbed.values)

            y_pred = np.squeeze(np.asarray(predictions))

            explanation = {
                "feature_name": feature_name,
                "feature_value": ordered_values,
                "ice_value": y_pred,
                "current_value": curr_value,
            }

        else:  # GroupedCE
            x_sample = pd.DataFrame(
                [
                    instance.values[0]
                    for _ in range(self.n_samples * self.n_samples + 1)
                ],
                columns=self.feature_names,
            )

            explanation = {"selected_features": self.features_selected}
            for i, feat_1 in enumerate(self.features_selected):
                if feat_1 not in explanation:
                    explanation[feat_1] = {}
                for feat_2 in self.features_selected[i + 1 :]:
                    feat_value_1, feat_value_2 = np.meshgrid(
                        feature_sample[feat_1], feature_sample[feat_2]
                    )
                    x_perturbed = x_sample.copy()
                    x_perturbed[feat_1] = np.append(
                        feat_value_1.reshape(-1), [instance[feat_1].values[0]]
                    )
                    x_perturbed[feat_2] = np.append(
                        feat_value_2.reshape(-1), [instance[feat_2].values[0]]
                    )
                    predictions = self._batch_predict(x_perturbed.values)

                    predictions = np.squeeze(np.asarray(predictions))
                    value_grid = predictions[:-1].reshape(
                        (self.n_samples, self.n_samples)
                    )
                    explanation[feat_1][feat_2] = {
                        "gce_values": value_grid.tolist(),
                        "x_grid": feature_sample[feat_1],
                        "y_grid": feature_sample[feat_2],
                    }
                    if return_instances:
                        explanation[feat_1][feat_2]["current_values"] = {
                            feat_1: instance[feat_1].values[0],
                            feat_2: instance[feat_2].values[0],
                        }
                        explanation[feat_1][feat_2]["prediction"] = predictions[-1]
        return explanation
