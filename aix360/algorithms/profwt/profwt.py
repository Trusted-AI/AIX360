from aix360.algorithms.gwbe import GlobalWBExplainer
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from aix360.algorithms.profwt.sample_weight_from_probes import prof_weight_compute

class ProfweightExplainer(GlobalWBExplainer):
    """
    The main explainer class that implements functions for computation of sample-wise weights from probe confidences of different layers from
    the complex model and then retrains the simple model with those weights. Implements the technique in the following reference: [#]_.

    References:
        .. [#] `Dhurandhar, Shanmugam, Luss, Olsen. Improving Simple Models with Confidence Profiles. NeurIPS 2018 <https://arxiv.org/abs/1807.07506>`_

    """
    def __init__(self):

        """
        Initialize ProfweightExplainer.
        """
        super(ProfweightExplainer, self).__init__()
        #self.model_args = model_args
        self._model = None

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        print("TBD: Implement set params in DIPVAEExplainer")

    def explain(self,x_train,y_train,x_test,y_test,simple_model,hps,list_probe_filenames,start_layer,end_layer,model_type='neural_keras'):
        """
        Obtains (train) sample-wise weights from stored probe confidences for different layers of the complex moddel. Retrains a simple
        model using the corresponding weighted training set.

        Parameters:
            x_train (numpy array): Dataset of features to retrain the simple model using weights derived from probe confidences. Dimensions (num of samples x feature dimensions)
            y_train (numpy array): Labels for the dataset that is used to retrain simple model using weights. Dimensions (num of samples x num of classes)
            x_test (numpy array): Test dataset to evaluate the simple model trained using weights. Dimensions (num of samples x feature dimensions)
            y_test (numpy array): Test labels to evaulate the simple model trained using weights. Dimensions (num of samples x num of classes)
            hps (namedtuple): Hyperparameters (usually a named tuple with entries) to train the simple model. Please see fit function to
                find out the required set of parameters expected by the fit function.
            list_probe_filenames (list of strings): List of strings indicated path to files where different probe confidences are stored.
            start_layer (int): Index corresponding to the starting layer whose probe confidences are going to be averaged to obtain weights.
                This is an index of list_probe_filenames.
            end_layer (int): Index corresponding to the last layer whose probe confidences are going to be averaged to obtain weights. This is an index of list_probe_filenames.
            model_type (string): This specifies the type of simple model to be trained. Default is 'neural_keras'. Only this option is implemented now.
            simple_model (function object for a Keras model): This is a function object that would initialize a keras model to specify the architecture of the simple model.

        Returns:
            None

        """


        w=prof_weight_compute(list_probe_filenames,start_layer,end_layer,y_train)
        # Train the simple model with Weights
        assert hasattr(hps,'checkpoint_path'), "Checkpoint to save not mentioned"
        m=self.fit(x_train,y_train,x_test,y_test,simple_model,hps,'neural_keras',w.reshape(w.shape[0],))
        print("Accuracy of Prof-Weighted Simple Model:",m[1])
        self._model=m[0]
        return


    def fit(self,x_train,y_train,x_test,y_test,simple_model,hps,model_type='neural_keras',sample_weight=None):
        """
        Fits the training data by initializing a simple model with hyper parameters and returns the test accuracy on the test dataset.
        This can be trained with or without sample weights. The first 500 samples of the test dataset is used as validation data.

        Parameters:
            x_train (numpy array): Training dataset features for training the simple model. Dimensions (num of samples x feature dimensions)
            y_train (numpy array): Labels for the training dataset to train on. Dimensions (num of samples x num of classes)
            x_test (numpy array): Test dataset features. Dimensions (num of samples x feature dimensions)
            y_test (numpy array): Test dataset labels. Dimensions (num of samples x num of classes)
            hps (namedtuple): A namedtuple that is expected to have the following named tuple elements:

                * optimizer - specified the optimizer in keras.
                * complexity_param - Used for Resenet based simple model to specify number of Resunits. Used by simple model function
                  object to intialize a simple model of appropriate complexity.
                * num_classes - scalar specifying number of classes used by the simple
                  model function.
                * checkpoint_path - specifies the path for saving a checkpoint of the trained model. This is expected.
                * lr_scheduler - a function object that takes in a scalar (epochs) and specified a learning rate (scalar). This is a learning rate Scheduler. Expected.
                * lr_reducer  - a function object that specifies how learning rates must be reduced if validation accuracy does not improve - Optional.

            simple_model (function object for a Keras model): A function object that constructs a keras model for the simple model and returns the model object. It is expected to take in input_shape, hps.complexity_param and num_classes.
                It is expected to implement a keras model fit function. It is also expected to implement a keras model evaulate function.

        Returns:
            tuple:

                * **model_d** (`Keras model object`) -- Returns the trained model that is initialized by simple_model functions.
                * **scores[1]** (`float`) -- Returns the test accuracy of the trained model on (x_test,y_test.)

        """


        input_shape=x_train.shape[1:]
        if model_type=='neural_keras':
            assert None not in (hps.complexity_param,hps.num_classes), "Missing Hyper Parameters"
            model_d=simple_model(input_shape,hps.complexity_param,hps.num_classes)
            assert hps.optimizer is not None, "Missing Optimizer Specs"
            model_d.compile(loss='categorical_crossentropy',optimizer=hps.optimizer,metrics=['accuracy'])
            if (hasattr(model_d,'summary')):
                model_d.summary()
            callbacks=[]
            assert hps.checkpoint_path is not None, "Checkpoint to save Model not specified"
            checkpoint = ModelCheckpoint(filepath=hps.checkpoint_path,monitor='val_acc',verbose=1,save_best_only=True)
            assert hps.lr_scheduler is not None, "Learning Rate Scheduler not specified"
            callbacks=callbacks+[checkpoint,hps.lr_scheduler]
            if hasattr(hps,'lr_reducer'):
                callbacks=callbacks+[hps.lr_reducer]
            assert hasattr(model_d,'fit'), "Model supplied needs a fit function"
            if sample_weight is not None:
                model_d.fit(x_train, y_train,batch_size=hps.batch_size,epochs=hps.epochs,validation_data=(x_test[0:500,:], y_test[0:500,:]),shuffle=True,callbacks=callbacks,sample_weight=sample_weight)
            else:
                model_d.fit(x_train, y_train,batch_size=hps.batch_size,epochs=hps.epochs,validation_data=(x_test[0:500,:], y_test[0:500,:]),shuffle=True,callbacks=callbacks)
            assert hasattr(model_d,'evaluate'), "Model supplied needs an evaluate function"
            scores=model_d.evaluate(x_test, y_test, verbose=1)

        return (model_d,scores[1])
