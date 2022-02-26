from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import tensorflow as tf
from aix360.algorithms.sif.SIF import SIFExplainer
from aix360.datasets.SIF_dataset import DataSet


class AllAR(SIFExplainer):
    def __init__(self, x_dim, y_dim, time_steps, share_param, **kwargs):
        self.time_steps = time_steps
        self.x_dim = x_dim
        self.cells = None
        self.y_dim = y_dim
        self.share_param = share_param
        if share_param:
            self.out_weights = tf.Variable(tf.random_normal([self.time_steps, 1]))
        else:
            self.out_weights = tf.Variable(tf.random_normal([self.time_steps, self.y_dim]))
        super().__init__(**kwargs)

    def get_all_params(self):
        all_params = []
        all_params.append(self.out_weights)
        return all_params

    def retrain(self, num_steps, feed_dict):
        retrain_dataset = DataSet(feed_dict[self.input_index_placeholder], feed_dict[self.labels_index_placeholder])
        for step in range(num_steps):
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)

    def placeholder_inputs(self):
        input_index_placeholder = tuple([tf.placeholder(
            tf.int32,
            shape=(None, 1),
            name='input_index_placeholder_{}'.format(i)) for i in range(self.time_steps)])
        labels_index_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, 1),
            name='labels_index_placeholder')
        ts_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, self.y_dim],
            name='input_ts')
        return input_index_placeholder, labels_index_placeholder, ts_placeholder

    def inference(self, input_x, labels_placeholder=None, keep_probs_placeholder=None):
        if self.share_param:
            weight = tf.tile(self.out_weights, [1, self.y_dim], name='Weight')
        else:
            weight = self.out_weights
        x = tf.stack([x[:, 0] for x in input_x], axis=1, name='x')
        y_hat = tf.einsum('ijk,jk->ik', x, weight, name='y_hat')
        return y_hat

    def predictions(self, logits):
        preds = logits
        return preds
    

class AllLSTM(SIFExplainer):
    def __init__(self, x_dim, y_dim, time_steps, num_units, share_param, **kwargs):
        self.time_steps = time_steps
        self.x_dim = x_dim
        self.num_units = num_units
        self.cells = None
        self.y_dim = y_dim
        self.share_param = share_param
        if share_param:
            self.out_weights = tf.Variable(tf.random_normal([self.num_units, 1]))
            self.out_bias = tf.Variable(tf.random_normal([1, 1]))
        else:
            self.out_weights = tf.Variable(tf.random_normal([self.num_units, self.y_dim]))
            self.out_bias = tf.Variable(tf.random_normal([1, self.y_dim]))
        super().__init__(**kwargs)

    def get_all_params(self):
        all_params = []
        lstm_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LSTM")
        all_params += lstm_variables
        all_params.append(self.out_weights)
        all_params.append(self.out_bias)
        return all_params

    def retrain(self, num_steps, feed_dict):
        retrain_dataset = DataSet(feed_dict[self.input_index_placeholder], feed_dict[self.labels_index_placeholder])
        for step in range(num_steps):
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)


    def placeholder_inputs(self):
        input_index_placeholder = tuple([tf.placeholder(
            tf.int32,
            shape=(None, 1),
            name='input_index_placeholder_{}'.format(i)) for i in range(self.time_steps)])
        labels_index_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, 1),
            name='labels_index_placeholder')
        ts_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, self.y_dim],
            name='input_ts')
        return input_index_placeholder, labels_index_placeholder, ts_placeholder

    def inference(self, input_x, labels_placeholder=None, keep_probs_placeholder=None):
        if isinstance(input_x, list) | isinstance(input_x, tuple):
            n = input_x[0].shape[2]
            x = [tuple(x0[:, :, i] for x0 in input_x) for i in range(n)]
        else:
            n = input_x.shape[2]
            x = [input_x[:, :, i] for i in range(n)]
        with tf.variable_scope("LSTM") as vs:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, name='LSTM_Layer')
            def run_lstm(x_n):
                output, _ = tf.nn.static_rnn(cell, x_n, dtype=tf.float32)
                return tf.matmul(output[-1], self.out_weights) + self.out_bias
            res = tf.stack(list(map(run_lstm, x)), axis=1)[:, :, 0]
        return res

    def predictions(self, logits):
        preds = logits
        return preds


class AllRNN(SIFExplainer):
    def __init__(self, x_dim, y_dim, time_steps, num_units, share_param, **kwargs):
        self.time_steps = time_steps
        self.x_dim = x_dim
        self.num_units = num_units
        self.cells = None
        self.y_dim = y_dim
        self.share_param = share_param
        if share_param:
            self.out_weights = tf.Variable(tf.random_normal([self.num_units, 1]))
            self.out_bias = tf.Variable(tf.random_normal([1, 1]))
        else:
            self.out_weights = tf.Variable(tf.random_normal([self.num_units, self.y_dim]))
            self.out_bias = tf.Variable(tf.random_normal([1, self.y_dim]))
        super().__init__(**kwargs)

    def get_all_params(self):
        all_params = []
        rnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="RNN")
        all_params += rnn_variables
        all_params.append(self.out_weights)
        all_params.append(self.out_bias)
        return all_params

    def retrain(self, num_steps, feed_dict):
        retrain_dataset = DataSet(feed_dict[self.input_index_placeholder], feed_dict[self.labels_index_placeholder])
        for step in range(num_steps):
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)

    def placeholder_inputs(self):
        input_index_placeholder = tuple([tf.placeholder(
            tf.int32,
            shape=(None, 1),
            name='input_index_placeholder_{}'.format(i)) for i in range(self.time_steps)])
        labels_index_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, 1),
            name='labels_index_placeholder')
        ts_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, self.y_dim],
            name='input_ts')
        return input_index_placeholder, labels_index_placeholder, ts_placeholder

    def inference(self, input_x, labels_placeholder=None, keep_probs_placeholder=None):
        from tensorflow.keras import layers
        model = tf.keras.Sequential()
        model.add(layers.Embedding(input_dim=1000, output_dim=64))
        # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
        model.add(layers.GRU(256, return_sequences=True))
        # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
        model.add(layers.SimpleRNN(128))
        model.add(layers.Dense(10, activation='softmax'))
        model.summary()
        if isinstance(input_x, list) | isinstance(input_x, tuple):
            n = input_x[0].shape[2]
            x = [tuple(x0[:, :, i] for x0 in input_x) for i in range(n)]
        else:
            n = input_x.shape[2]
            x = [input_x[:, :, i] for i in range(n)]
        with tf.variable_scope("RNN") as vs:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, name='RNN_Layer')
            def run_rnn(x_n):
                output, _ = tf.nn.static_rnn(cell, x_n, dtype=tf.float32)
                return tf.matmul(output[-1], self.out_weights) + self.out_bias

            res = tf.stack(list(map(run_rnn, x)), axis=1)[:, :, 0]
        return res

    def predictions(self, logits):
        preds = logits
        return preds
