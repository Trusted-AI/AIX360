""" Tests for :py:mod:`aix360.algorithms.nncontrastive.NearestNeighborContrastiveExplainer`.
"""
# disable tensorflow warnings
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from aix360.algorithms.nncontrastive import NearestNeighborContrastiveExplainer

warnings.filterwarnings("ignore")


class TestNearestNeighborContrastiveExplainer(unittest.TestCase):
    def setUp(self):

        # load data
        dataset = load_breast_cancer()
        X = dataset["data"]
        y = dataset["target"]

        feature_names = dataset["feature_names"]

        xx_train, xx_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )

        min_x = np.min(xx_train, axis=0)
        max_x = np.max(xx_train, axis=0)

        self.xx_train = (xx_train - min_x[np.newaxis, :]) / (
            max_x[np.newaxis, :] - min_x[np.newaxis, :]
        )
        self.xx_train = pd.DataFrame(self.xx_train, columns=feature_names)

        self.xx_test = (xx_test - min_x[np.newaxis, :]) / (
            max_x[np.newaxis, :] - min_x[np.newaxis, :]
        )
        self.xx_test = pd.DataFrame(self.xx_test, columns=feature_names)

        self.y_train_onehot = np.concatenate(
            [1 - y_train[:, np.newaxis], y_train[:, np.newaxis]], axis=1
        ).astype("float32")
        self.y_test_onehot = np.concatenate(
            [1 - y_test[:, np.newaxis], y_test[:, np.newaxis]], axis=1
        ).astype("float32")

        # train example model
        self.model = RandomForestClassifier(n_estimators=2, random_state=10)
        self.model.fit(self.xx_train, y_train)

    def test_single_instance_only(self):

        # initialize/fit explainer
        epochs = 500
        embedding_dim = 4
        layers_config = []
        random_seed = 1
        neighbors = 5
        explainer = NearestNeighborContrastiveExplainer(
            embedding_dim=embedding_dim,
            layers_config=layers_config,
            neighbors=neighbors,
        )

        benign_exemplars = self.xx_train.iloc[np.where(self.y_train_onehot == 1)[0], :]

        explainer.fit(
            self.xx_train,
            epochs=epochs,
            numeric_scaling=None,
            exemplars=benign_exemplars,
            random_seed=random_seed,
        )

        y_t_pred = np.round(self.model.predict(self.xx_test)).astype(int)

        xx_test_benign = self.xx_test.iloc[np.where(y_t_pred.reshape(-1) == 1)[0], :]

        # compute explanations
        explanation = explainer.explain_instance(xx_test_benign.iloc[0])

        # validate explanation structure
        self.assertIn("features", explanation)
        self.assertIn("categorical_features", explanation)
        self.assertIn("query", explanation)
        self.assertIn("neighbors", explanation)
        self.assertIn("distances", explanation)
        self.assertEquals(len(explanation["neighbors"]), neighbors)
        self.assertEquals(len(explanation["distances"]), neighbors)

    def test_single_instance_with_model(self):

        # initialize/fit explainer
        epochs = 500
        embedding_dim = 4
        layers_config = []
        random_seed = 1
        neighbors = 3

        explainer = NearestNeighborContrastiveExplainer(
            model=self.model.predict,
            embedding_dim=embedding_dim,
            layers_config=layers_config,
            neighbors=neighbors,
        )

        explainer.fit(
            self.xx_train,
            epochs=epochs,
            numeric_scaling=None,
            random_seed=random_seed,
        )

        explanation = explainer.explain_instance(self.xx_test.iloc[10])

        # validate explanation structure
        self.assertIn("features", explanation)
        self.assertIn("categorical_features", explanation)
        self.assertIn("query", explanation)
        self.assertIn("neighbors", explanation)
        self.assertIn("distances", explanation)
        self.assertLessEqual(len(explanation["neighbors"]), neighbors)
        self.assertLessEqual(len(explanation["distances"]), neighbors)

        prediction = self.model.predict(np.asarray([self.xx_test.iloc[10]])).reshape(
            -1
        )[0]
        prediction = list(np.array([prediction], dtype=int).reshape(-1))[0]

        neighbor_prediction = self.model.predict(
            np.asarray([explanation["neighbors"][0]])
        )
        neighbor_prediction = list(
            np.array([neighbor_prediction], dtype=int).reshape(-1)
        )[0]

        # validate that the explanation is contrastive
        self.assertNotEqual(prediction, neighbor_prediction)

    def test_multiple_instances(self):

        # initialize/fit explainer
        epochs = 500
        embedding_dim = 4
        layers_config = []
        random_seed = 1
        neighbors = 3

        explainer = NearestNeighborContrastiveExplainer(
            embedding_dim=embedding_dim,
            layers_config=layers_config,
            neighbors=neighbors,
        )

        benign_exemplars = self.xx_train.iloc[np.where(self.y_train_onehot == 1)[0], :]

        explainer.fit(
            self.xx_train,
            epochs=epochs,
            numeric_scaling=None,
            random_seed=random_seed,
        )

        y_t_pred = np.round(self.model.predict(self.xx_test.values)).astype(int)
        xx_test_benign = self.xx_test.iloc[np.where(y_t_pred.reshape(-1) == 1)[0], :]

        # compute explanations
        explainer.set_exemplars(benign_exemplars)
        explanations = explainer.explain_instance(xx_test_benign.iloc[0:5])

        # validate explanation structure
        self.assertEqual(len(explanations), 5)
        explanation = explanations[0]
        self.assertIn("features", explanation)
        self.assertIn("categorical_features", explanation)
        self.assertIn("query", explanation)
        self.assertIn("neighbors", explanation)
        self.assertIn("distances", explanation)
        self.assertEquals(len(explanation["neighbors"]), neighbors)
        self.assertEquals(len(explanation["distances"]), neighbors)


if __name__ == "__main__":
    unittest.main()
