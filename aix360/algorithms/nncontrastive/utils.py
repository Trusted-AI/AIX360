import math
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import StratifiedShuffleSplit

class AESampler(Sequence):
    def __init__(self, 
                 x, 
                 batch_size: int = 512,
                 max_size: int = 10000,
                 validation_f: float = 0.1,
                 ):
        self._x = x 
        self._max_size = max_size 
        self._validation_size = min(int(max_size*validation_f), 
                                    self._x.shape[0] - self._max_size)
        self._index = np.arange(x.shape[0])
        self._batch_size = batch_size
    
    def on_epoch_end(self):
        np.random.shuffle(self._index)
        
    def __len__(self):
        return math.ceil(self._max_size / self._batch_size)
                
    def __getitem__(self, idx):
        batch_start =  idx * self._batch_size
        batch_end = min((idx + 1) * self._batch_size, self._max_size)
        dataX = self._x[self._index[batch_start:batch_end]]
        return dataX, dataX
            

class JointAESampler(AESampler):
    def __init__(self,
                 x, 
                 y,
                 batch_size: int = 512,
                 max_size: int = 10000, 
                 validation_f: float = 0.1,
                 ):
        super(JointAESampler, self).__init__(x, 
                                             batch_size,
                                             max_size, 
                                             validation_f)
        self._y = y 
        self._counter = 0
        self._n_splits = min(x.shape[0] // max_size, 10)
        self._splits = []
        
    def on_epoch_end(self):
        self._counter = 0
        sss = StratifiedShuffleSplit(n_splits=self._n_splits, 
                                     train_size=self._max_size,
                                     test_size=self._validation_size)
        self._splits = sss.split(self._x, self._y)
    
    def __len__(self):
        return self._n_splits
    
    def __getitem__(self, idx):
        train_index, _ = self._splits[idx]
        dataX = self._x[train_index]
        dataY = self._y[train_index]
        return dataX, [dataX, dataY]
    
