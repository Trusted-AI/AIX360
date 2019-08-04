import numpy as np
import json
import sys,os


class CIFARDataset():
    """
    The CIFAR-10 dataset [#]_ consists of 60000 32x32 color images. Target variable is one amongst 10 classes. The dataset has
    6000 images per class. There are 50000 training images and 10000 test images. The classes are: airplane, automobile,
    bird, cat, deer, dog, frog, horse, ship ,truck. We further divide the training set into train1 (30000 samples) and
    train2 (20,000 samples). For ProfWt, the complex model is trained on train1 while the simple model is trained on train2.

    References:
        .. [#] `Krizhevsky, Hinton. Learning multiple layers of features from tiny images. Technical Report, University of
           Toronto 1 (4), 7. 2009 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    """

    def __init__(self, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','cifar_data')

    def load_file(self, filename):
        try:
            with open(os.path.join(self._dirpath, filename)) as file:
                data=json.load(file)
            file.close()
        except IOError as err:
            print("IOError: {}".format(err))
            sys.exit(1)
        return np.array(data)
