import unittest
from aix360.datasets import DiabetesDataset
from sklearn.ensemble import RandomForestRegressor
from aix360.algorithms.gce.gce import GroupedCEExplainer


class TestGroupedCEExplainer(unittest.TestCase):
    def setUp(self):

        # load data
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.feature_names,
            self.target_names,
        ) = DiabetesDataset().load_data(
            test_size=0.2, random_state=42, return_only_numerical=True
        )
        self.clf = RandomForestRegressor().fit(self.x_train, self.y_train)

    def test_ice(self):

        # load model

        n_samples = 10
        ice_explainer = GroupedCEExplainer(
            model=self.clf.predict,
            data=self.x_train,
            feature_names=self.feature_names,
            n_samples=n_samples,
            features_selected=["BMI"],
            random_seed=22,
        )
        explanation = ice_explainer.explain_instance(instance=self.x_test[[0], :])
        # validate explanation structure
        self.assertIn("feature_name", explanation)
        self.assertIn("feature_value", explanation)
        self.assertIn("ice_value", explanation)
        self.assertIn("current_value", explanation)

        self.assertEqual(explanation["ice_value"].shape[0], n_samples)

    def test_gce(self):

        # load model

        n_samples = 10
        top_k_features = 4
        gce_explainer = GroupedCEExplainer(
            model=self.clf.predict,
            data=self.x_train,
            feature_names=self.feature_names,
            n_samples=n_samples,
            top_k_features=top_k_features,
            random_seed=22,
        )
        explanation = gce_explainer.explain_instance(instance=self.x_test[[0], :])
        # validate explanation structure

        self.assertIn("selected_features", explanation)
        self.assertEquals(len(explanation["selected_features"]), top_k_features)
        self.assertTrue(
            set(explanation["selected_features"]).issubset(set(explanation.keys()))
        )
        feat_1 = explanation["selected_features"][0]
        feat_2 = explanation["selected_features"][1]
        self.assertIn(feat_2, explanation[feat_1])
        self.assertIn("gce_values", explanation[feat_1][feat_2])
        self.assertIn("x_grid", explanation[feat_1][feat_2])
        self.assertIn("y_grid", explanation[feat_1][feat_2])
