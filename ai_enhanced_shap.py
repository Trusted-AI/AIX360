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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TestShapExplainer(unittest.TestCase):

    def test_Shap(self):
        np.random.seed(1)
        X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

        # K-nearest neighbors
        knn = sklearn.neighbors.KNeighborsClassifier()
        knn.fit(X_train, Y_train)

        # AI-driven feature: Automatically calculate additional performance metrics
        Y_pred = knn.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='macro')
        recall = recall_score(Y_test, Y_pred, average='macro')
        f1 = f1_score(Y_test, Y_pred, average='macro')

        print(f"Accuracy = {accuracy * 100:.2f}%")
        print(f"Precision = {precision:.2f}")
        print(f"Recall = {recall:.2f}")
        print(f"F1 Score = {f1:.2f}")

        # Explain a single prediction from the test set
        shapexplainer = KernelExplainer(knn.predict_proba, X_train)
        shap_values = shapexplainer.explain_instance(X_test.iloc[0,:])  # AI-driven: Debugging output
        print('knn X_test iloc_0 SHAP values:', shap_values)

        # AI-driven feature: Enhanced visualization for SHAP values
        shap.summary_plot(shap_values, X_test)

        # SV machine with a linear kernel
        svc_linear = sklearn.svm.SVC(kernel='linear', probability=True)
        svc_linear.fit(X_train, Y_train)

        # Calculate additional metrics for SVC
        Y_pred_svc = svc_linear.predict(X_test)
        svc_accuracy = accuracy_score(Y_test, Y_pred_svc)
        svc_precision = precision_score(Y_test, Y_pred_svc, average='macro')
        svc_recall = recall_score(Y_test, Y_pred_svc, average='macro')
        svc_f1 = f1_score(Y_test, Y_pred_svc, average='macro')

        print(f"SVC Accuracy = {svc_accuracy * 100:.2f}%")
        print(f"SVC Precision = {svc_precision:.2f}")
        print(f"SVC Recall = {svc_recall:.2f}")
        print(f"SVC F1 Score = {svc_f1:.2f}")

        # Explain all the predictions in the test set
        shapexplainer = KernelExplainer(svc_linear.predict_proba, X_train)
        shap_values = shapexplainer.explain_instance(X_test)
        print('svc X_test SHAP values:', shap_values)

        # Enhanced visualization
        shap.summary_plot(shap_values, X_test)

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

        # AI-driven feature: Performance metrics for the linear model
        Y_pred = model.predict(X_test)
        linear_accuracy = accuracy_score(y_test, Y_pred)
        linear_precision = precision_score(y_test, Y_pred, average='macro')
        linear_recall = recall_score(y_test, Y_pred, average='macro')
        linear_f1 = f1_score(y_test, Y_pred, average='macro')

        print(f"Linear Model Accuracy = {linear_accuracy * 100:.2f}%")
        print(f"Linear Model Precision = {linear_precision:.2f}")
        print(f"Linear Model Recall = {linear_recall:.2f}")
        print(f"Linear Model F1 Score = {linear_f1:.2f}")

        # Enhanced SHAP visualization
        shap.summary_plot(shap_values, X_test)

    # comment this test as travis runs out of resources
    def test_ShapGradientExplainer(self):
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

        # Enhanced visualization for image explanations
        shap.image_plot(shap_values, x_test[1:5])

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
        model_full = xgboost.train(params, xgb_full, 10000, evals=[(xgb_full, "test")], verbose_eval=1000)

        explainer = shap.TreeExplainer(model_full)
        shap_values = explainer.shap_values(X)

        # AI-driven feature: Enhanced interpretation of Tree SHAP values
        print("Tree SHAP values:", shap_values)
        shap.summary_plot(shap_values, X_display)

if __name__ == '__main__':
    unittest.main()
