# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets


class DataSetTS():
    def __init__(self,
                 y:np.ndarray,
                 y_name:np.ndarray = None,
                 t: np.ndarray = None,
                 x: np.ndarray = None,
                 x_name:np.ndarray = None,
                 split_ratio = (0.6, 0.2, 0.2),
                 lag = 1,
                 scaler = None,
                 reference_ts = None
                 ):

        # check all the dimensions:
        n_t = y.shape[0]
        if len(y.shape) != 2:
            y = np.reshape(y, (n_t, -1))

        n_y = y.shape[1]

        if y_name is not None:
            assert y_name.shape[0] == n_y, "y_name must agree with the dimension of y"

        if t is not None:
            assert t == n_t, "t must be the same length with x"

        if x is not None and len(x.shape) != 2:
            x = np.reshape(x, [x.shape[0], -1])
            assert x.shape[0] == n_t, "x must have the same length with x"

            if x_name is not None:
                x_name = np.reshape(x_name, -(1))

            n_f = x.shape[0]
            if x_name is not None:
                assert x_name.shape[0] == n_f, "x_name must agree with the dimension of x"
        else:
            n_f = 0

        # record all the information
        self.x = x
        self.x_name = x_name
        self.y = y
        self.y_name = y_name
        self.t = t

        self.n_t = n_t
        self.n_f = n_f

        self.split_ratio = split_ratio
        self.lag = lag

        self.scaler = scaler

        self.split_train_test()
        self.reference_ts = reference_ts

    def moving_win_idx(self, init_win_len, horizon, moving_step, fixed_win, n_t):
        stops = np.arange(init_win_len, n_t - horizon, moving_step)
        if fixed_win:
            starts = stops - init_win_len
        else:
            starts = np.repeat(0, stops.shape[0])
        train = map(np.arange, starts, stops)
        test = map(np.arange, stops, stops+horizon)
        return train, test, stops-1

    def split_train_test(self, split_ratio=None):
        if split_ratio is None:
            split_ratio = self.split_ratio
        else:
            self.split_ratio = split_ratio

        hist_idx, pred_idx, t_idx = self.moving_win_idx(self.lag, 1, 1, True, self.n_t)

        all_steps = list() # y_n, y_(0:n), x_n, t_n
        for tr, te, tt in zip(hist_idx, pred_idx, t_idx):
            all_steps.append((self.y[te, :], self.y[tr, :],
                             self.x[tr, :] if self.x is not None else None,
                             self.t[tt] if self.t is not None else tt))

        n_train = int(self.n_t*split_ratio[0])

        """
        if self.batch_size is not None: # adjust the n_train and n_validate to fit the batch_size
            n_train = round(n_train/self.batch_size) * self.batch_size
            # n_valid = round(n_valid/self.batch_size) * self.batch_size
            assert n_train > 0, "the batch_size is too large"
            # assert n_valid > 0, "the batch_size is too large"
        """
        self.train = all_steps[:n_train]
        self.test = all_steps[(n_train):]

    def _split_y_x(self, dt):
        y = np.vstack([s[0] for s in dt])
        if self.x is None:
            x = np.vstack([np.reshape(s[1], (1, s[1].shape[0], -1)) for s in dt])
        else:
            x = np.vstack([np.reshape(np.hstack((s[1], s[2])), (1, s[1].shape[0], -1)) for s in dt])
        return tuple([x[:, i] for i in range(x.shape[1])]), y


    def datasets_gen_rnn(self):
        """
        Generate a x, y pair ready to feed into RNN
        :return:
            x - a three dimensional array: sample x time steps x feature
            y - a two dimensional array: sample x 1
        """
        train = DataSet(*(self._split_y_x(self.train)), reference_ts=self.reference_ts)
        test = DataSet(*(self._split_y_x(self.test)), reference_ts=self.reference_ts)

        return Datasets(train=train, validation=test, test=test)


class DataSet(object):

    def __init__(self, x, labels, reference_ts = None):
        if isinstance(x, list) or isinstance(x, tuple):
            if len(x[0].shape) > 2:
                x = [np.reshape(i.astype(np.int), [i.shape[0], -1]) for i in x]
            else:
                x = [i.astype(np.int) for i in x]
            assert (x[0].shape[0] == labels.shape[0])

        else:
            if len(x.shape) > 2:
                x = np.reshape(x, [x.shape[0], -1])
            assert (x[0].shape[0] == labels.shape[0])
            x = x.astype(np.int)

        self._num_examples = labels.shape[0]

        self._x = x
        self._labels = labels

        # used for the output of index to make sure there is no nan goes into the calculation
        self.reference_ts = reference_ts


    @property
    def x(self):
        if self._nonnan_idx is None:
            return self._x
        if isinstance(self._x, list):
            return [x[self._nonnan_idx] for x in self._x]
        return self._x[self._nonnan_idx]

    @property
    def labels(self):
        if self._nonnan_idx is None:
            return self._labels
        return self._labels[self._nonnan_idx]

    @property
    def reference_ts(self):
        return self._reference_ts

    @reference_ts.setter
    def reference_ts(self, value):
        if value is None:
            self._reference_ts = None
            self._nonnan_idx = None

            if isinstance(self._x, list):
                self._x_batch = [np.copy(ele) for ele in self.x]
            else:
                self._x_batch = np.copy(self.x)
        else:
            self._reference_ts = value
            if isinstance(self._x, list):
                if value is not None:  # check if reference ts contains nan
                    self._nonnan_idx = np.sum(np.hstack([np.isnan(value[x[:, 0], ...]) for x in self._x]
                                               + [np.isnan(value[self._labels[:, 0], ...])]), axis=1) == 0
                    self._x_batch = [np.copy(ele) for ele in self.x]
            else:
                if value is not None:  # check if reference ts contains nan
                    self._nonnan_idx = np.sum(np.hstack((np.isnan(value[self._x, ...]),
                                                np.isnan(value[self._labels[:, 0], ...]))), axis=1) == 0
                    self._x_batch = np.copy(self.x)

        self._labels_batch = np.copy(self.labels)
        self._index_in_epoch = 0

    @property
    def num_examples(self):
        if self._nonnan_idx is None:
            return self._num_examples
        else:
            return sum(self._nonnan_idx)

    def reset_batch(self):
        self._index_in_epoch = 0
        if isinstance(self._x_batch, list):
            self._x_batch = [np.copy(ele) for ele in self.x]
        else:
            self._x_batch = np.copy(self.x)

        self._labels_batch = np.copy(self.labels)

    def next_batch(self, batch_size):
        assert batch_size <= self.num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self.num_examples:

            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            if isinstance(self._x_batch, list):
                self._x_batch = [i[perm,:] for i in self._x_batch]
            else:
                self._x_batch = self._x_batch[perm, :]
            self._labels_batch = self._labels_batch[perm, :]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        if isinstance(self._x_batch, list):
            xx = [ele[start:end] for ele in self._x_batch]
            yy = self._labels_batch[start:end, :]

        else:
            xx = self._x_batch[start:end]
            yy = self._labels_batch[start:end, :]
        #assert not np.any(np.vstack(np.isnan([self.reference_ts[x] for x in xx]))), ""
        #assert not np.any(np.vstack(np.isnan([self.reference_ts[x] for x in yy]))), ""
        return xx, yy


def filter_dataset(X, Y, pos_class, neg_class):
    """
    Filters out elements of X and Y that aren't one of pos_class or neg_class
    then transforms labels of Y so that +1 = pos_class, -1 = neg_class.
    """
    assert(X.shape[0] == Y.shape[0])
    assert(len(Y.shape) == 1)

    Y = Y.astype(int)
    
    pos_idx = Y == pos_class
    neg_idx = Y == neg_class        
    Y[pos_idx] = 1
    Y[neg_idx] = -1
    idx_to_keep = pos_idx | neg_idx
    X = X[idx_to_keep, ...]
    Y = Y[idx_to_keep]    
    return (X, Y)    


def find_distances(target, X, theta=None):
    assert len(X.shape) == 2, "X must be 2D, but it is currently %s" % len(X.shape)
    target = np.reshape(target, -1)
    assert X.shape[1] == len(target), \
      "X (%s) and target (%s) must have same feature dimension" % (X.shape[1], len(target))
    
    if theta is None:
        return np.linalg.norm(X - target, axis=1)
    else:
        theta = np.reshape(theta, -1)
        
        # Project onto theta
        return np.abs((X - target).dot(theta))
