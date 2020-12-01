import abc
import sys
import numpy as np
import torch

# compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class BaseClassifier(ABC):

    """
    Base class for classifiers (image, tabular data, etc.)
    The base class is meant to be framework independent so that each
    framework specific classes (keras, torch, etc.) could inherit and
    implement these methods.
    """
    def __init__(self):

        """
        Initialize a Classifier object.
        """

    @abc.abstractmethod
    def predict(self, x, verbose=0):
        """
        Prediction on a batch of inputs x.
        type: numpy.ndarray
        """
        raise NotImplementedError


    @property
    def nb_classes(self):
        """
        Returns the number of output classes
        """
        return self._nb_classes

    @property
    def input_shape(self):
        """
        Returns the shape of one input to the classifier
        """
        return self._input_shape


class KerasClassifier(BaseClassifier):
    """Keras classifier has the following attributes:

    Attributes:
        _model: the compiled keras classifier
        _input:  input placeholder of the model
        _output: output placeholder of the model
        _nb_classes:  number of outputs (e.g. binary classifier has 1 output)
        _input_shape: shape of 1 input sample
    """
    def __init__(self, model, input_layer=0, output_layer=0):

        """Initialize KerasClassifier.

        Args:
            model: a trained keras classifier model.
        """

        import keras.backend as k

        super(KerasClassifier, self).__init__()

        self._model = model

        if hasattr(model, 'inputs'):
            self._input = model.inputs[input_layer]
        else:
            self._input = model.input

        if hasattr(model, 'outputs'):
            self._output = model.outputs[output_layer]
        else:
            self._output = model.output

        _, self._nb_classes = k.int_shape(self._output)
        self._input_shape   = k.int_shape(self._input)[1:]

    def predict(self, x, verbose=0):
        """
        Make predictions on batch of vector inputs
        """
        return(self._model.predict(x, verbose=verbose))

    def predict_classes(self, x, verbose=0):
        """
        returns classes instead of probabilities
        """
        return(self._model.predict_classes(x, verbose=verbose))

    def predict_long(self, x):
        prob = self.predict(x)
        predicted_class = np.argmax(prob)
        prob_str = np.array2string(prob).replace('\n','')
        return(prob, predicted_class, prob_str)

    def predictsym(self, x):
        return self._model(x)


class PytorchClassifier():

    def __init__(self, model):

        self._model = model

        self._nb_classes = model.num_classes
        self._input_shape = model.input_shape

    def predict(self, x, verbose=0):
        """
        Make predictions on batch of vector inputs
        """
        return(self._model(x))

    def predict_classes(self, x, verbose=0):
        """
        returns classes instead of probabilities
        """
        logits = self.predict(x)
        predicted_class = [torch.argmax(logits).item()]

        return(predicted_class)

    def predict_long(self, x):
        logits = self.predict(x)
        predicted_class = torch.argmax(logits).item()
        logits_np = logits.detach().cpu().numpy()
        logits_str = np.array2string(logits_np).replace('\n','')
        return(logits_np, predicted_class, logits_str)

    def predictsym(self, x):
        return self._model(x)
