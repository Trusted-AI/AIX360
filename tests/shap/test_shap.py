from __future__ import print_function
import unittest

import sklearn
import sklearn.datasets
import sklearn.ensemble

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K
import json

import xgboost

from aix360.algorithms.shap import KernelExplainer, LinearExplainer, GradientExplainer, DeepExplainer, TreeExplainer
import shap

class TestShapExplainer(unittest.TestCase):

    def test_Shap(self):

        np.random.seed(1)
        X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

        # K-nearest neighbors
        knn = sklearn.neighbors.KNeighborsClassifier()
        knn.fit(X_train, Y_train)
        v = 100*np.sum(knn.predict(X_test) == Y_test)/len(Y_test)
        print("Accuracy = {0}%".format(v))

        # Explain a single prediction from the test set
        shapexplainer = KernelExplainer(knn.predict_proba, X_train)
        shap_values = shapexplainer.explain_instance(X_test.iloc[0,:])  # TODO test against original SHAP Lib
        print('knn X_test iloc_0')
        print(shap_values)
        print(shapexplainer.explainer.expected_value[0])
        print(shap_values[0])

        # Explain all the predictions in the test set
        shap_values = shapexplainer.explain_instance(X_test)
        print('knn X_test')
        print(shap_values)
        print(shapexplainer.explainer.expected_value[0])
        print(shap_values[0])

        # SV machine with a linear kernel
        svc_linear = sklearn.svm.SVC(kernel='linear', probability=True)
        svc_linear.fit(X_train, Y_train)
        v = 100*np.sum(svc_linear.predict(X_test) == Y_test)/len(Y_test)
        print("Accuracy = {0}%".format(v))

        # Explain all the predictions in the test set
        shapexplainer = KernelExplainer(svc_linear.predict_proba, X_train)
        shap_values = shapexplainer.explain_instance(X_test)
        print('svc X_test')
        print(shap_values)
        print(shapexplainer.explainer.expected_value[0])
        print(shap_values[0])

        np.random.seed(1)
        X,y = shap.datasets.adult()
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=7)

        knn = sklearn.neighbors.KNeighborsClassifier()
        knn.fit(X_train, y_train)

        f = lambda x: knn.predict_proba(x)[:,1]
        med = X_train.median().values.reshape((1,X_train.shape[1]))
        shapexplainer = KernelExplainer(f, med)
        shap_values_single = shapexplainer.explain_instance(X.iloc[0,:], nsamples=1000)
        print('Shap Tabular Example')
        print(shapexplainer.explainer.expected_value)
        print(shap_values_single)
        print("Invoked Shap KernelExplainer")


    def test_ShapLinearExplainer(self):
        corpus, y = shap.datasets.imdb()
        corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=7)

        vectorizer = TfidfVectorizer(min_df=10)
        X_train = vectorizer.fit_transform(corpus_train)
        X_test = vectorizer.transform(corpus_test)

        model = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1, solver='liblinear')
        model.fit(X_train, y_train)

        shapexplainer = LinearExplainer(model, X_train, feature_dependence="independent")
        shap_values = shapexplainer.explain_instance(X_test)
        print("Invoked Shap LinearExplainer")

    # comment this test as travis runs out of resources
    def test_ShapGradientExplainer(self):

    #     model = VGG16(weights='imagenet', include_top=True)
    #     X, y = shap.datasets.imagenet50()
    #     to_explain = X[[39, 41]]
    #
    #     url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    #     fname = shap.datasets.cache(url)
    #     with open(fname) as f:
    #         class_names = json.load(f)
    #
    #     def map2layer(x, layer):
    #         feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    #         return K.get_session().run(model.layers[layer].input, feed_dict)
    #
    #     e = GradientExplainer((model.layers[7].input, model.layers[-1].output),
    #                           map2layer(preprocess_input(X.copy()), 7))
    #     shap_values, indexes = e.explain_instance(map2layer(to_explain, 7), ranked_outputs=2)
    #
          print("Skipped Shap GradientExplainer")


    def test_ShapDeepExplainer(self):
        batch_size = 128
        num_classes = 10
        epochs = 2

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # select a set of background examples to take an expectation over
        background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

        # explain predictions of the model on three images
        e = DeepExplainer(model, background)

        shap_values = e.explain_instance(x_test[1:5])
        print("Invoked Shap DeepExplainer")


    def test_ShapTreeExplainer(self):
        X, y = shap.datasets.nhanesi()
        X_display, y_display = shap.datasets.nhanesi(display=True)  # human readable feature values

        xgb_full = xgboost.DMatrix(X, label=y)

        # create a train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        xgb_train = xgboost.DMatrix(X_train, label=y_train)
        xgb_test = xgboost.DMatrix(X_test, label=y_test)

        # use validation set to choose # of trees
        params = {
            "eta": 0.002,
            "max_depth": 3,
            "objective": "survival:cox",
            "subsample": 0.5
        }
        model_train = xgboost.train(params, xgb_train, 10000, evals=[(xgb_test, "test")], verbose_eval=1000)

        # train final model on the full data set
        params = {
            "eta": 0.002,
            "max_depth": 3,
            "objective": "survival:cox",
            "subsample": 0.5
        }
        model = xgboost.train(params, xgb_full, 5000, evals=[(xgb_full, "test")], verbose_eval=1000)

        def c_statistic_harrell(pred, labels):
            total = 0
            matches = 0
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if labels[j] > 0 and abs(labels[i]) > labels[j]:
                        total += 1
                        if pred[j] > pred[i]:
                            matches += 1
            return matches / total

        # see how well we can order people by survival
        c_statistic_harrell(model_train.predict(xgb_test, ntree_limit=5000), y_test)

        shap_values = TreeExplainer(model).explain_instance(X)
        print("Invoked Shap TreeExplainer")


if __name__ == '__main__':
    unittest.main()
