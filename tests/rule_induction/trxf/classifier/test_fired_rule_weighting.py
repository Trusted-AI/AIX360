"""Real TRXF rule metrics and classifier; requires installed pandas/numpy/sklearn.

Namespace containers bypass unrelated optional top-level AIX360 initializers.
No feature, predicate, metric, dataframe, aggregation or inference is mocked.
"""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def load_trxf(source_root):
    root = Path(source_root)
    packages = {}
    for name in ("aix360", "aix360.algorithms", "aix360.algorithms.rule_induction",
                 "aix360.algorithms.rule_induction.trxf",
                 "aix360.algorithms.rule_induction.trxf.core",
                 "aix360.algorithms.rule_induction.trxf.classifier"):
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        packages[name] = module
    loaded = {}
    with patch.dict(sys.modules, packages):
        for relative in ("core/boolean_evaluator", "core/feature", "core/predicate",
                         "core/utils", "core/conjunction", "core/dnf_ruleset",
                         "metrics", "classifier/ruleset_classifier"):
            name = "aix360.algorithms.rule_induction.trxf." + relative.replace("/", ".")
            path = root / "aix360/algorithms/rule_induction/trxf" / (relative + ".py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[relative] = module
    return loaded


class FiredRuleWeightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("AIX_SOURCE_ROOT")
        if root is None:
            target = Path("aix360/algorithms/rule_induction/trxf/classifier/ruleset_classifier.py")
            for parent in Path(__file__).resolve().parents:
                for location in (parent, parent / "candidate"):
                    if (location / target).is_file():
                        root = location
                        break
                if root is not None:
                    break
            if root is None:
                raise FileNotFoundError("Cannot locate the AIX360 source tree")
        cls.modules = load_trxf(root)
        cls.api = cls.modules["classifier/ruleset_classifier"]

    def conjunction(self, feature, relation, value):
        Feature = self.modules["core/feature"].Feature
        Predicate = self.modules["core/predicate"].Predicate
        return self.modules["core/conjunction"].Conjunction([Predicate(Feature(feature), relation, value)])

    def ruleset(self, label, *conjunctions):
        return self.modules["core/dnf_ruleset"].DnfRuleSet(list(conjunctions), label)

    def classifier(self, method, rulesets, X, y, default=None, laplace=False):
        api = self.api
        result = api.RuleSetClassifier(
            rulesets, method,
            confidence_metric=api.ConfidenceMetric.LAPLACE if laplace else api.ConfidenceMetric.CONSTANT,
            weight_metric=api.WeightMetric.CONFIDENCE if laplace else api.WeightMetric.CONSTANT,
            default_label=default,
        )
        # All weighted data tests initialize actual weights through real metrics.
        result.update_rules_with_metrics(pd.DataFrame(X), pd.Series(y))
        return result

    def methods(self):
        return (self.api.RuleSelectionMethod.WEIGHTED_SUM, self.api.RuleSelectionMethod.WEIGHTED_MAX)

    def signed_rules(self):
        relation = self.modules["core/predicate"].Relation
        return [self.ruleset("positive", self.conjunction("x", relation.GT, 0)),
                self.ruleset("negative", self.conjunction("x", relation.LT, 0))]

    def test_different_assignments_select_different_firing_labels(self):
        X = {"x": np.array([-2., -1., 1., 2.])}
        for method in self.methods():
            with self.subTest(method=method):
                classifier = self.classifier(method, self.signed_rules(), X,
                                             ["negative", "negative", "positive", "positive"])
                self.assertEqual([r.weight for r in classifier.rules], [1, 1])
                self.assertEqual(classifier.predict({"x": 1}), "positive")
                self.assertEqual(classifier.predict({"x": -1}), "negative")

    def test_unfired_high_weight_cannot_defeat_fired_lower_weight(self):
        for method in self.methods():
            with self.subTest(method=method):
                classifier = self.classifier(method, self.signed_rules(),
                    {"x": [-2, -1, 1, 2]}, ["negative", "negative", "negative", "positive"], laplace=True)
                self.assertAlmostEqual(classifier.rules[0].weight, 0.5)
                self.assertAlmostEqual(classifier.rules[1].weight, 0.75)
                self.assertEqual(classifier.predict({"x": 1}), "positive")

    def test_sum_and_max_keep_distinct_weight_aggregation(self):
        relation = self.modules["core/predicate"].Relation
        rules = [self.ruleset("A", self.conjunction("x", relation.GT, 0),
                             self.conjunction("y", relation.GT, 0)),
                 self.ruleset("B", self.conjunction("z", relation.GT, 0))]
        X = {"x": [1, 1, 1, 1, -1], "y": [1, 1, 1, 1, -1], "z": [-1, -1, -1, 1, 1]}
        y = ["A", "A", "A", "B", "B"]
        summed = self.classifier(self.methods()[0], rules, X, y, laplace=True)
        maximum = self.classifier(self.methods()[1], rules, X, y, laplace=True)
        for classifier in (summed, maximum):
            self.assertAlmostEqual(classifier.rules[0].weight, 2 / 3)
            self.assertAlmostEqual(classifier.rules[1].weight, 2 / 3)
            self.assertAlmostEqual(classifier.rules[2].weight, 0.75)
        assignment = {"x": 1, "y": 1, "z": 1}
        self.assertEqual(summed.predict(assignment), "A")
        self.assertEqual(maximum.predict(assignment), "B")

    def test_no_firing_rule_returns_configured_default(self):
        for method in self.methods():
            for default in (None, "unknown", 0, False, ""):
                with self.subTest(method=method, default=default):
                    classifier = self.classifier(method, self.signed_rules(), {"x": [-1, 1]},
                                                 ["negative", "positive"], default=default)
                    self.assertIs(classifier.predict({"x": 0}), default)

    def test_empty_rule_collections_return_default(self):
        for method in self.methods():
            for rulesets in ([], [self.ruleset("unused")]):
                with self.subTest(method=method, rulesets=rulesets):
                    marker = object()
                    classifier = self.classifier(method, rulesets, {"x": [-1, 1]},
                                                 ["negative", "positive"], default=marker)
                    self.assertEqual(classifier.rules, [])
                    self.assertIs(classifier.predict({}), marker)

    def test_existing_first_label_tie_order_is_preserved(self):
        relation = self.modules["core/predicate"].Relation
        rules = [self.ruleset("z-first", self.conjunction("x", relation.GT, 0)),
                 self.ruleset("a-second", self.conjunction("x", relation.GT, 0))]
        for method in self.methods():
            with self.subTest(method=method):
                classifier = self.classifier(method, rules, {"x": [-1, 1]}, ["a-second", "z-first"])
                self.assertEqual(classifier.predict({"x": 1}), "z-first")

    def test_first_hit_does_not_need_weights_and_keeps_early_winner(self):
        relation = self.modules["core/predicate"].Relation
        classifier = self.api.RuleSetClassifier([
            self.ruleset("first", self.conjunction("x", relation.GT, 0)),
            self.ruleset("second", self.conjunction("x", relation.GT, -1)),
        ], self.api.RuleSelectionMethod.FIRST_HIT, default_label="none")
        self.assertEqual(classifier.predict({"x": 1}), "first")
        self.assertEqual(classifier.predict({"x": 0}), "second")
        self.assertEqual(classifier.predict({"x": -2}), "none")
        self.assertTrue(all(r.weight is None for r in classifier.rules))

    def test_nonempty_weighted_models_keep_uninitialized_weight_guard(self):
        for method in self.methods():
            with self.subTest(method=method):
                classifier = self.api.RuleSetClassifier(self.signed_rules(), method, default_label="none")
                for x in (1, -1):
                    with self.assertRaises(AssertionError):
                        classifier.predict({"x": x})

    def test_zero_weight_firing_label_does_not_fall_back(self):
        relation = self.modules["core/predicate"].Relation
        for method in self.methods():
            with self.subTest(method=method):
                classifier = self.api.RuleSetClassifier([
                    self.ruleset("A", self.conjunction("x", relation.GT, 0))
                ], method, confidence_metric=self.api.ConfidenceMetric.ACCURACY,
                   weight_metric=self.api.WeightMetric.CONFIDENCE, default_label="none")
                classifier.update_rules_with_metrics(pd.DataFrame({"x": [-1, 1]}), pd.Series(["A", "B"]))
                self.assertEqual(classifier.rules[0].weight, 0)
                self.assertEqual(classifier.predict({"x": 1}), "A")

    def test_empty_conjunction_is_a_true_rule_not_an_empty_rule_collection(self):
        conjunction = self.modules["core/conjunction"].Conjunction([])
        for method in self.methods():
            with self.subTest(method=method):
                classifier = self.classifier(method, [self.ruleset("always", conjunction)],
                    {"x": [-1, 1]}, ["always", "other"], default="none")
                self.assertEqual(classifier.predict({}), "always")

    def test_missing_feature_error_is_not_swallowed_as_no_match(self):
        for method in self.methods():
            with self.subTest(method=method):
                classifier = self.classifier(method, self.signed_rules(),
                    {"x": [-1, 1]}, ["negative", "positive"], default="none")
                with self.assertRaises(ValueError):
                    classifier.predict({})

    def test_invalid_selection_method_with_empty_rules_still_raises(self):
        classifier = self.api.RuleSetClassifier([], "not-a-selection-method", default_label="none")
        with self.assertRaisesRegex(ValueError, "Unknown rule selection method"):
            classifier.predict({})


if __name__ == "__main__":
    unittest.main()
