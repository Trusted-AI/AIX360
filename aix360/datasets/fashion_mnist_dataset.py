import numpy as np
#import torch.nn as nn
import torch
import os
#import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
#from torch.autograd import Variable
#from torch.nn import Linear



class FMnistDataset():
    """
    Fashion-MNIST [#]_ is a large-scale image dataset of various fashion items (T-shirt/top, Trouser,
    Pullover, Dress. Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. The images are grayscale and 28x28
    in size with each image belong to one the above mentioned 10 categories. The training set contains
    60000 examples and the test set contains 10000 examples.

    References:
        .. [#] `Xiao, Han, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a
           novel image dataset for benchmarking machine learning algorithms.
           <https://arxiv.org/abs/1708.07747>`_
    """
    def __init__(self, batch_size=256, subset_size=50000, test_batch_size=256, dirpath=None):
        trans = transforms.Compose([transforms.ToTensor()])

        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','fmnit_data')


        train_set = dset.FashionMNIST(root=self._dirpath, train=True, transform=trans, download=True)
        test_set = dset.FashionMNIST(root=self._dirpath, train=False, transform=trans, download=True)

        indices = torch.randperm(len(train_set))[:subset_size]
        train_set = torch.utils.data.Subset(train_set, indices)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=test_batch_size,
            shuffle=False)

        self.name = "fmnist"
        self.data_dims = [28, 28, 1]
        self.train_size = len(self.train_loader)
        self.test_size = len(self.test_loader)
        self.range = [0.0, 1.0]
        self.batch_size = batch_size
        self.num_training_instances = len(train_set)
        self.num_test_instances = len(test_set)
        self.likelihood_type = 'gaussian'
        self.output_activation_type = 'sigmoid'

    def next_batch(self):
        for x, y in self.train_loader:
            x = np.reshape(x, (-1, 28, 28, 1))
            yield x, y

    def next_test_batch(self):
        for x, y in self.test_loader:
            x = np.reshape(x, (-1, 28, 28, 1))
            yield x, y
