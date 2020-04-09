## based on setup_mnist.py -- mnist data and model loading code
##
## Original Work Copyright (c) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the "supplementary license" folder present in the root directory.
## 
## Modifications Copyright (c) 2019 IBM Corporation


import numpy as np
import os
import gzip
import urllib.request


    
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data
    
def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)
    
class MNISTDataset():
    def __init__(self, custom_preprocessing=None, dirpath=None): 
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','mnist_data')

        files = ["train-images-idx3-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz",
                 "train-labels-idx1-ubyte.gz",
                 "t10k-labels-idx1-ubyte.gz"]
        for name in files:
            if not os.path.exists(self._dirpath + "/" + name):
                print("retrieving file", name)
                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, self._dirpath + "/" + name)
                print("retrieved")

        train_data       = extract_data(self._dirpath + "/train-images-idx3-ubyte.gz", 60000)
        train_labels     = extract_labels(self._dirpath + "/train-labels-idx1-ubyte.gz", 60000)
        self.test_data   = extract_data(self._dirpath + "/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels(self._dirpath + "/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]
        
        
