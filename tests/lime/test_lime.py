import unittest
import os
import shutil

import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from skimage.color import gray2rgb, rgb2gray, label2rgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from lime.wrappers.scikit_image import SegmentationAlgorithm

from aix360.algorithms.lime import LimeTabularExplainer
from aix360.algorithms.lime import LimeTextExplainer
from aix360.algorithms.lime import LimeImageExplainer


class TestLIMEExplainer(unittest.TestCase):

    def test_LIME(self):

        # test invocation of lime explainer on tabular data
        iris = sklearn.datasets.load_iris()
        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data,
                                                                                          iris.target,
                                                                                          train_size=0.80)
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
        rf.fit(train, labels_train)

        sklearn.metrics.accuracy_score(labels_test, rf.predict(test))

        explainer = LimeTabularExplainer(train,
                                     feature_names=iris.feature_names,
                                     class_names=iris.target_names,
                                     discretize_continuous=True)

        i = 19
        explanation = explainer.explain_instance(test[i], rf.predict_proba, num_features=2, top_labels=1)
        print(i, explanation.as_map())
        print('Invoked Tabular explainer\n')

        # test invocation of lime explainer on text data

        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')

        # making class names shorter
        class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in
                       newsgroups_train.target_names]
        class_names[3] = 'pc.hardware'
        class_names[4] = 'mac.hardware'

        print(','.join(class_names))

        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
        train_vectors = vectorizer.fit_transform(newsgroups_train.data)
        test_vectors = vectorizer.transform(newsgroups_test.data)

        nb = MultinomialNB(alpha=.01)
        nb.fit(train_vectors, newsgroups_train.target)

        pred = nb.predict(test_vectors)
        sklearn.metrics.f1_score(newsgroups_test.target, pred, average='weighted')

        c = make_pipeline(vectorizer, nb)
        print(c.predict_proba([newsgroups_test.data[0]]).round(3))

        explainer = LimeTextExplainer(class_names=class_names)

        idx = 1340
        exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6, labels=[0, 17])
        print('Document id: %d' % idx)
        print('Predicted class =', class_names[nb.predict(test_vectors[idx]).reshape(1, -1)[0, 0]])
        print('True class: %s' % class_names[newsgroups_test.target[idx]])

        print('Explanation for class %s' % class_names[0])
        print('\n'.join(map(str, exp.as_list(label=0))))
        print()
        print('Explanation for class %s' % class_names[17])
        print('\n'.join(map(str, exp.as_list(label=17))))

        print('Invoked Text explainer\n')


        # test invocation of lime explainer on Image data
        mnist = fetch_openml('mnist_784')

        # make each image color so lime_image works correctly
        X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))], 0)
        y_vec = mnist.target.astype(np.uint8)

        class PipeStep(object):
            """
            Wrapper for turning functions into pipeline transforms (no-fitting)
            """

            def __init__(self, step_func):
                self._step_func = step_func

            def fit(self, *args):
                return self

            def transform(self, X):
                return self._step_func(X)

        makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
        flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

        simple_rf_pipeline = Pipeline([
            ('Make Gray', makegray_step),
            ('Flatten Image', flatten_step),
            # ('Normalize', Normalizer()),
            # ('PCA', PCA(16)),
            ('RF', RandomForestClassifier())
        ])

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec, train_size=0.55)
        simple_rf_pipeline.fit(X_train, y_train)

        explainer = LimeImageExplainer(verbose=False)
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

        explanation = explainer.explain_instance(X_test[0],
                                                 classifier_fn=simple_rf_pipeline.predict_proba,
                                                 top_labels=10, hide_color=0, num_samples=10000,
                                                 segmentation_fn=segmenter)
        print('Invoked Image explainer\n')

if __name__ == '__main__':
    unittest.main()