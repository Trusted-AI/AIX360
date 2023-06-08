import unittest

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from aix360.algorithms.imd.utils import load_bc_dataset
from aix360.algorithms.imd.rule import Rule
from aix360.algorithms.imd.imd import IMDExplainer
from aix360.algorithms.imd.utils import visualize_jst


class TestIMDExplainer(unittest.TestCase):

    def test_rule_class(self):
        rule1 = Rule(0, [('x1', '<=', 6), ('x2', '<=', 100), ('x1', '>', 1)], 0)
        rule2 = Rule(1, [('x1', '<=', 9), ('x2', '>', 100), ('x1', '>', 3)], 1)
        rule3 = Rule(2, [('x1', '<=', 2), ('x2', '<=', 90)], 1)

        rule31 = Rule(31, [('x2', '<=', 90), ('x1', '>', 1), ('x1', '<=', 2)], 1)
        rule2_ = Rule(100, [('x1', '>', 3), ('x2', '>', 100), ('x1', '<=', 9)], -1)

        self.assertTrue(rule31.check_equal_preds(rule1.intersection(rule3)), "check rule1 intersection rule3 predicate equivalence")
        self.assertTrue(rule2.check_equal_preds(rule2_), "check rule2 and rule2_ predicate equivalence")

        self.assertEqual(rule1.as_dict(),
                         {'x1': [1, 6], 'x2': [np.NINF, 100]}
                         )
        self.assertEqual(rule2.as_dict(),
                         {'x1': [3, 9], 'x2': [100, np.inf]}
                         )
        self.assertEqual(rule3.as_dict(),
                         {'x1': [np.NINF, 2], 'x2': [np.NINF, 90]}
                         )
        self.assertEqual(rule3.intersection(rule1).as_dict(),
                         {'x1': [1, 2], 'x2': [np.NINF, 90]}
                         )
        self.assertEqual(rule1.intersection(rule2).as_dict(), {})

    def test_imd(self):
        random_state = 1234
        datadf, target = load_bc_dataset()
        x_train, x_test, y_train, y_test = train_test_split(datadf, target, train_size=0.7,
                                                            random_state=random_state)

        self.assertEqual(x_train.shape[0], 398)
        self.assertEqual(x_test.shape[1], 30)

        ## model1
        model1 = DecisionTreeClassifier(max_depth=5)
        model1.fit(x_train, y_train)
        # print(f"model: {model1}")
        tacc = accuracy_score(y_true=y_test, y_pred=model1.predict(x_test))
        self.assertGreater(tacc, 0.9)
        # print(f"model1 test accuracy: {(tacc * 100):.2f}%")

        ## model2
        model2 = GaussianNB()
        model2.fit(x_train, y_train)

        # print(f"model: {model2}")
        tacc = accuracy_score(y_true=y_test, y_pred=model2.predict(x_test))
        self.assertGreater(tacc, 0.85)
        # print(f"model2 test accuracy: {(tacc * 100):.2f}%")

        feature_names = x_train.columns.to_list()
        x1 = x2 = x_train.to_numpy()
        y1 = model1.predict(x1)
        y2 = model2.predict(x2)
        ydiff = (y1 != y2).astype(int)
        # print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff)):.2f}")

        ydifftest = (model1.predict(x_test) != model2.predict(x_test)).astype(int)
        # print(f"diffs in X_test = {ydifftest.sum()} / {len(ydifftest)} = {(ydifftest.sum() / len(ydifftest)):.2f}")

        max_depth = 6
        imd = IMDExplainer()
        imd.fit(x_train, y1, y2, max_depth=max_depth, verbose=False)

        diffrules = imd.explain()
        testmetrics = imd.metrics(x_test, model1.predict(x_test), model2.predict(x_test), name="test")
        imagepath = visualize_jst(imd.jst, path="joint.jpg")


if __name__ == '__main__':
    unittest.main()