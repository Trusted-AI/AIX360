from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
import pandas as pd
import copy
from scipy.optimize import fmin_ncg
import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K
import collections
from aix360.algorithms.sif.SIF_utils import hessian_vector_product, hessians, operate_deep, operate_deep_2v, normalize_vector
from abc import ABC, abstractmethod
from aix360.datasets.SIF_dataset import DataSet
from scipy.stats import linregress
from aix360.algorithms.lbbe import LocalBBExplainer


class SIFExplainer(LocalBBExplainer):
    def __init__(self, **kwargs):
        '''
        Initialize the SIF neural network
        '''
        np.random.seed(0)
        tf.set_random_seed(0)
        self.Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
        self.batch_size = kwargs.pop('batch_size')
        self.time_series = kwargs.pop('time_series')
        self.data_sets = kwargs.pop('data_sets')
        self.train_dir = kwargs.pop('train_dir', 'output')
        self.log_dir = kwargs.pop('log_dir', 'log')
        self.model_name = kwargs.pop('model_name')
        # self.num_classes = kwargs.pop('num_classes')
        self.initial_learning_rate = kwargs.pop('initial_learning_rate')
        self.decay_epochs = kwargs.pop('decay_epochs')
        self.calc_hessian = kwargs.pop('calc_hessian')

        if 'keep_probs' in kwargs:
            self.keep_probs = kwargs.pop('keep_probs')
        else:
            self.keep_probs = None

        if 'mini_batch' in kwargs:
            self.mini_batch = kwargs.pop('mini_batch')
        else:
            self.mini_batch = True

        if 'damping' in kwargs:
            self.damping = kwargs.pop('damping')
        else:
            self.damping = 0.0

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # Initialize session
        config = tf.compat.v1.ConfigProto()
        self.sess = tf.compat.v1.Session(config=config)

        # TODO: Remove me after debugging finishes
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        K.set_session(self.sess)

        # Setup input
        self.input_index_placeholder, self.labels_index_placeholder, self.ts_placeholder = self.placeholder_inputs()
        self.num_train_examples = self.data_sets.train.labels.shape[0]
        self.num_test_examples = self.data_sets.test.labels.shape[0]

        if isinstance(self.input_index_placeholder, list) or isinstance(self.input_index_placeholder, tuple):
            self.input_placeholder = tuple(tf.gather(self.ts_placeholder, x) for x in self.input_index_placeholder)
        else:
            self.input_placeholder = tf.gather(self.ts_placeholder, self.input_index_placeholder,
                                               name='input_placeholder')
        self.labels_placeholder = tf.reshape(tf.gather(self.ts_placeholder, self.labels_index_placeholder),
                                             [-1, self.ts_placeholder.shape[1].value], name='label_placeholder')

        # Setup inference and training
        if self.keep_probs is not None:
            self.keep_probs_placeholder = tf.placeholder(tf.float32, shape=(2))
            self.logits = self.inference(self.input_placeholder, keep_probs_placeholder=self.keep_probs_placeholder)
        elif hasattr(self, 'inference_needs_labels'):
            self.logits = self.inference(self.input_placeholder, labels_placeholder=self.labels_placeholder)
        else:
            self.logits = self.inference(self.input_placeholder)

        self.total_loss, self.loss_no_reg, self.indiv_loss_no_reg = self.loss(
            self.logits,
            self.labels_placeholder)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(self.initial_learning_rate, name='learning_rate', trainable=False)
        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.update_learning_rate_op = tf.assign(self.learning_rate, self.learning_rate_placeholder)
        # self.learning_rate = 0.01
        self.train_op = self.get_train_op(self.total_loss, self.global_step, self.learning_rate)
        self.train_sgd_op = self.get_train_sgd_op(self.total_loss, self.global_step, self.learning_rate)
        # self.accuracy_op = self.get_accuracy_op(self.logits, self.labels_placeholder)
        self.preds = self.predictions(self.logits)

        # Setup misc
        self.saver = tf.train.Saver()

        # Setup gradients and Hessians
        self.params = self.get_all_params()
        # \nabla_\theta L
        self.grad_total_loss_op = tf.gradients(self.total_loss, self.params)
        # remove the independent parameters
        self.params, self.grad_total_loss_op = zip(
            *[(a, b) for a, b in zip(self.params, self.grad_total_loss_op) if b is not None])
        # \nabla_\theta L: no regularizer
        self.grad_loss_no_reg_op = tf.gradients(self.loss_no_reg, self.params)
        # u,v: a list the same size \theta
        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape(), name='v_placeholder') for a in
                              self.params]

        # H \cdot v
        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)

        # \nabla_x L
        self.grad_loss_wrt_input_op = tf.gradients(self.total_loss, self.ts_placeholder)
        # \nabla_\theta L \cdots v
        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)
        self.influence_op = tf.add_n(
            [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b)))
             for a, b in zip(self.grad_total_loss_op, self.v_placeholder)])
        # (\nabla_x\nabla_\theta L) \codts v: assuming v not depend on x
        # self.grad_influence_wrt_input_op = tf.gradients(self.influence_op, self.ts_placeholder)
        self.subgradient = tf.gradients(self.total_loss, self.params[0])
        if self.calc_hessian:
            self.hessian = hessians(self.total_loss, self.params)

        # self.grad_loss_wrt_para_input_op = derivative_x1_x2(self.total_loss, self.params,
        #                                                    self.ts_placeholder)

        self.checkpoint_file = os.path.join(self.train_dir, "%s-checkpoint" % self.model_name)

        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)
        # self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)
        self.optimizer = tf.compat.v1.train.MomentumOptimizer(0.007, momentum=0.95)
        # self.optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
        # self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(0.001)
        self.t = tf.Variable(tf.random.normal(self.v_placeholder[0].shape), name='t', trainable=True)
        # self.hessian_vector_1 = hessian_vector_product(self.total_loss, self.params, [self.t])
        # self.hessian_vector_val = tf.add(self.hessian_vector_1, self.t * self.damping)
        # fun = 0.5 * tf.matmul(tf.transpose(self.t), self.hessian_vector_val) - tf.matmul(
        #     tf.transpose(self.v_placeholder[0]), self.t)

        # ---------  test  ----------
        self.hessian = hessians(self.total_loss, self.params)[0]
        # self.hessian_vector_val = tf.add(tf.matmul(self.hessian[0], self.t), self.t * self.damping)
        fun = 0.5 * tf.matmul(tf.transpose(self.t), tf.matmul(self.hessian[0], self.t)) - tf.matmul(
            tf.transpose(self.v_placeholder[0]), self.t)
        # ---------  test  ----------

        # self.training_process = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate).minimize(fun, var_list=[self.t])
        self.training_process = self.optimizer.minimize(fun, var_list=[self.t])
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.num_train_step = None
        self.iter_to_switch_to_batch = None
        self.iter_to_switch_to_sgd = None

        self.vec_to_list = self.get_vec_to_list_fn()
        self.adversarial_loss, self.indiv_adversarial_loss = self.adversarial_loss(self.logits, self.labels_placeholder)
        if self.adversarial_loss is not None:
            self.grad_adversarial_loss_op = tf.gradients(self.adversarial_loss, self.params)

        # newly added configure to speed up the computation
        self.n_time_stamp = self.time_series.shape[0]
        self.n_non_nan_y_cont = 0
        self.non_nan_y_cont_idx = []
        super().__init__()

    def set_params(self, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, y_contaminate, index, patchy_k, gammas=None, expectation_rep_time=10, verbose=True,
                         is_ar=False, short_memo=True):
        """
        Explain one or more input instances.
        """
        sif = self.get_sif(y_contaminate, index, patchy_k, gammas, expectation_rep_time, verbose, is_ar, short_memo)
        return sif

    def update_configure(self, y_contaminate, gammas):
        '''
        y_contaminate: the contaminating process
        gammas: gamma in equation 1
        '''
        self.n_non_nan_y_cont = self.n_time_stamp - np.isnan(y_contaminate).sum()
        self.rest_gamma = gammas[np.nonzero(gammas)]
        self.large_gamma = [x for x in self.rest_gamma if x * self.n_time_stamp >= self.n_non_nan_y_cont]
        self.rest_gamma = [x for x in self.rest_gamma if x * self.n_time_stamp < self.n_non_nan_y_cont]
        self.non_nan_y_cont_idx = np.logical_not(np.isnan(y_contaminate))

    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.params)

        self.num_params = len(np.concatenate([np.reshape(x, (-1, 1)) for x in params_val]))
        print('Total number of parameters: %s' % self.num_params)

        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(np.reshape(v[cur_pos: cur_pos + p.size], p.shape))
                cur_pos += p.size

            assert cur_pos == len(v)
            return return_list

        return vec_to_list

    def reset_datasets(self):
        '''
        reset the batch
        '''
        for data_set in self.data_sets:
            if data_set is not None:
                data_set.reset_batch()

    def fill_feed_dict_with_all_ex(self, data_set):
        '''
        return a dictionary which contains all examples
        '''
        feed_dict = {
            self.ts_placeholder: self.time_series,
            self.input_index_placeholder: data_set.x,
            self.labels_index_placeholder: data_set.labels
        }
        return feed_dict

    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):
        '''
        return a dictionary which contains all but one example (leave one out)
        data_set: the dataset
        idx_to_remove: the index that is not included in the returned dictionary
        '''
        num_examples = data_set.num_examples
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False

        if isinstance(data_set.x, list):
            input_x = [x[idx, :] for x in data_set.x]
        else:
            input_x = data_set[idx, :]

        feed_dict = {
            self.ts_placeholder: self.time_series,
            self.input_index_placeholder: input_x,
            self.labels_index_placeholder: data_set.labels[idx]
        }
        return feed_dict

    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        '''
        return a dictionary whose batch is equal to batch_size
        data_set: the dataset
        batch_size: the size of the batch to be returned
        '''
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size

        input_feed, labels_feed = data_set.next_batch(batch_size)

        feed_dict = {
            self.ts_placeholder: self.time_series,
            self.input_index_placeholder: input_feed,
            self.labels_index_placeholder: labels_feed
        }
        return feed_dict

    def fill_feed_dict_with_some_ex(self, data_set, target_indices, time_seres=None):
        '''
        return a dictionary whose batch is equal to batch_size
        data_set: the dataset
        target_indices: the indices of examples to be returned
        '''
        if isinstance(data_set.x, list):
            input_feed = [x[target_indices, :].reshape(len(target_indices), -1)
                          for x in data_set.x]
        else:
            input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices, :].reshape(len(target_indices), -1)

        feed_dict = {
            self.ts_placeholder: self.time_series if time_seres is None else time_seres,
            self.input_index_placeholder: input_feed,
            self.labels_index_placeholder: labels_feed
        }
        return feed_dict

    def fill_feed_dict_with_one_ex(self, data_set, target_idx, time_series=None):
        '''
        return a dictionary which contains one example
        data_set: the dataset
        target_idx: the index of an example to be returned
        '''
        if isinstance(data_set.x, list):
            input_x = [x[target_idx, :].reshape(1, -1) for x in data_set.x]
        else:
            input_x = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx, :].reshape(1, -1)
        feed_dict = {
            self.ts_placeholder: self.time_series if time_series is None else time_series,
            self.input_index_placeholder: input_x,
            self.labels_index_placeholder: labels_feed
        }
        return feed_dict

    def fill_feed_dict_with_one_perturb(self, data_set, target_idx, epsilon):
        '''
        return a dictionary which contains one example after perturbation
        data_set: the dataset
        target_idx: the index of an example to be returned
        '''
        if target_idx not in data_set.labels:
            if isinstance(data_set.x, (list, tuple)):
                assert any([target_idx in x for x in data_set.x]), "The index must be included in the dataset"
            else:
                assert target_idx in data_set.x, "The index must be included in the dataset"

        new_ts = np.copy(self.time_series)
        new_ts[target_idx] += epsilon
        feed_dict = {
            self.ts_placeholder: new_ts,
            self.input_index_placeholder: data_set.x,
            self.labels_index_placeholder: data_set.labels
        }
        return feed_dict

    def fill_feed_dict_manual(self, X, Y):
        '''
        return a dictionary with user-defined examples
        X: the input data
        Y: the label of input data
        '''
        X = np.array(X)
        Y = np.array(Y)
        input_feed = X.reshape(len(Y), -1)
        labels_feed = Y.reshape(-1)
        feed_dict = {
            self.ts_placeholder: self.time_series,
            self.input_index_placeholder: input_feed,
            self.labels_index_placeholder: labels_feed
        }
        return feed_dict

    def minibatch_mean_eval(self, ops, data_set):

        num_examples = data_set.num_examples
        # TODO: Let's think about this
        # assert num_examples % self.batch_size == 0
        num_iter = int(num_examples / self.batch_size)
        self.reset_datasets()
        ret = []
        for i in xrange(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(data_set)
            ret_temp = self.sess.run(ops, feed_dict=feed_dict)

            if len(ret) == 0:
                for b in ret_temp:
                    if isinstance(b, list) or isinstance(b, tuple):
                        ret.append([c / float(num_iter) for c in b])
                    else:
                        ret.append([b / float(num_iter)])
            else:
                for counter, b in enumerate(ret_temp):
                    if isinstance(b, list) or isinstance(b, tuple):
                        ret[counter] = [a + (c / float(num_iter)) for (a, c) in zip(ret[counter], b)]
                    else:
                        ret[counter] += (b / float(num_iter))

        return ret

    def get_train_y_hat(self):
        '''
        evaluate the y_hat with training examples
        '''
        return self.sess.run(self.preds, feed_dict=self.all_train_feed_dict)

    def get_test_y_hat(self):
        '''
        evaluate the y_hat with test examples
        '''
        return self.sess.run(self.preds, feed_dict=self.all_test_feed_dict)

    def print_model_eval(self):
        '''
        print the evaluation of the model related information, such as loss, gradients, norm of parameters, etc.
        '''
        params_val = self.sess.run(self.params)

        if self.mini_batch == True:
            grad_loss_val, loss_no_reg_val, loss_val = self.minibatch_mean_eval(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss],
                self.data_sets.train)

            test_loss_val = self.minibatch_mean_eval(
                [self.loss_no_reg],
                self.data_sets.test)

        else:
            grad_loss_val, loss_no_reg_val, loss_val = self.sess.run(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss],
                feed_dict=self.all_train_feed_dict)

            test_loss_val = self.sess.run(
                [self.loss_no_reg],
                feed_dict=self.all_test_feed_dict)

        print('Train loss (w reg) on all data: %s' % loss_val)
        print('Train loss (w/o reg) on all data: %s' % loss_no_reg_val)
        print('Test loss (w/o reg) on all data: %s' % test_loss_val)

        print('Norm of the mean of gradients: %s' % np.linalg.norm(
            np.concatenate([np.reshape(x, [-1, 1]) for x in grad_loss_val])))
        print('Norm of the params: %s' % np.linalg.norm(np.concatenate([np.reshape(x, [-1, 1]) for x in params_val])))

    def retrain(self, num_steps, feed_dict):
        '''
        retrain the model with num_steps iterations
        num_steps: the number of iterations
        feed_dict: the training examples
        '''
        for step in xrange(num_steps):
            self.sess.run(self.train_op, feed_dict=feed_dict)

    def update_learning_rate(self, step):
        '''
        update the learning rate
        step: when the step or iteration is reached, update the learning rate
        '''
        if self.mini_batch:
            assert self.num_train_examples % self.batch_size == 0
            num_steps_in_epoch = self.num_train_examples / self.batch_size
        else:
            num_steps_in_epoch = 1
        epoch = step // num_steps_in_epoch

        multiplier = 1
        if epoch < self.decay_epochs[0]:
            multiplier = 1
        elif epoch < self.decay_epochs[1]:
            multiplier = 0.1
        else:
            multiplier = 0.01

        self.sess.run(
            self.update_learning_rate_op,
            feed_dict={self.learning_rate_placeholder: multiplier * self.initial_learning_rate})

    def train(self, num_steps,
              iter_to_switch_to_batch=20000,
              iter_to_switch_to_sgd=40000,
              save_checkpoints=True, verbose=True,
              ):
        """
        Trains a model for a specified number of steps.
        num_steps: the number of iterations
        iter_to_switch_to_batch: the number of iterations to switch to batch training
        iter_to_switch_to_sgd: the number of iterations to switch to sgd optimizer
        save_checkpoints: Whether to save the model at the checkpoints
        verbose: whether to print the message during the training
        """
        if verbose: print('Training for %s steps' % num_steps)
        self.num_train_step = num_steps
        self.iter_to_switch_to_batch = iter_to_switch_to_batch
        self.iter_to_switch_to_sgd = iter_to_switch_to_sgd

        sess = self.sess

        # Tensorboard
        # train_writer = tf.summary.FileWriter('./logs/{}/train'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")),
        #                                    sess.graph)
        # tf.summary.scalar("loss", self.total_loss)
        # merged = tf.summary.merge_all()
        org_loss = -100
        err = []
        for step in range(num_steps):
            self.update_learning_rate(step)

            start_time = time.time()

            if step < iter_to_switch_to_batch:
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train)
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)
                # loss_val = sess.run([self.total_loss], feed_dict=feed_dict)

            elif step < iter_to_switch_to_sgd:
                feed_dict = self.all_train_feed_dict
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)

            else:
                feed_dict = self.all_train_feed_dict
                _, loss_val = sess.run([self.train_sgd_op, self.total_loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # train_writer.add_summary(summary, step)

            if (step + 1) % 1000 == 0:
                # Print status to stdout.
                if verbose:
                    print('Step %d: loss = %.8f (%.3f sec)' % (step, loss_val, duration))
                # if(abs(loss_val - org_loss) < epsilon):
                #    break
                # org_loss = loss_val
            err.append((step, loss_val))

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 10000 == 0 or (step + 1) == num_steps:
                if save_checkpoints: self.saver.save(sess, self.checkpoint_file, global_step=step)
                if verbose: self.print_model_eval()
        return err

    # train_writer.flush()

    def load_checkpoint(self, iter_to_load, do_checks=True):
        '''
        load the model at the checkpoint
        iter_to_load: the number of iteration where the model is saved
        do_checks: print the informaiton of the model after loading
        '''
        checkpoint_to_load = "%s_%s" % (self.checkpoint_file, iter_to_load)
        self.saver.restore(self.sess, checkpoint_to_load)

        if do_checks:
            print('Model %s loaded. Sanity checks ---' % checkpoint_to_load)
            self.print_model_eval()

    def save(self, file_name, do_checks=False):
        '''
        save the model at the checkpoint
        file_name: the name of the file with which the model is saved
        do_checks: print the information of the model after loading
        '''
        self.saver.save(self.sess, file_name)
        if do_checks:
            print('Model %s sSaved. Sanity checks ---' % file_name)
            self.print_model_eval()

    def restore(self, file_name, do_checks=False):
        '''
        load the model at the checkpoint
        file_name: the name of the file with which the model is saved
        do_checks: print the information of the model after loading
        '''
        self.saver.restore(self.sess, file_name)
        if do_checks:
            print('Model %s loaded. Sanity checks ---' % file_name)
            self.print_model_eval()

    def restore_train(self, file_name, init_step,
                      iter_to_switch_to_batch=20000,
                      iter_to_switch_to_sgd=40000,
                      ):
        """
        Trains a model for a specified number of steps.
        file_name: the name of the file with which the model is saved
        init_step: the threshold to train the model with different learning rate, different batches, etc.
        iter_to_switch_to_batch: the number of iterations to switch to batch training
        iter_to_switch_to_sgd: the number of iterations to switch to sgd optimizer
        """
        self.num_train_step = init_step
        self.iter_to_switch_to_batch = iter_to_switch_to_batch
        self.iter_to_switch_to_sgd = iter_to_switch_to_sgd

        sess = self.sess

        org_loss = -100
        err = []

        self.update_learning_rate(init_step)

        start_time = time.time()

        if init_step < iter_to_switch_to_batch:
            feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train)
            _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)
            # loss_val = sess.run([self.total_loss], feed_dict=feed_dict)

        elif init_step < iter_to_switch_to_sgd:
            feed_dict = self.all_train_feed_dict
            _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)

        else:
            feed_dict = self.all_train_feed_dict
            _, loss_val = sess.run([self.train_sgd_op, self.total_loss], feed_dict=feed_dict)

        duration = time.time() - start_time

        self.load_checkpoint(file_name, do_checks=True)
        # train_writer.add_summary(summary, step)
        print('inital loss = %.8f (%.3f sec)' % (loss_val))

    def get_train_op(self, total_loss, global_step, learning_rate):
        """
        Return train_op
        total_loss: the loss function to be optimized
        global_step: the global step for the optimizer
        learning_rate: the learning rate of the adam optimizer
        """
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step, )
        return train_op

    def get_train_sgd_op(self, total_loss, global_step, learning_rate=0.001):
        """
        Return train_sgd_op
        total_loss: the loss function to be optimized
        global_step: the global step for the optimizer
        learning_rate: the learning rate of the SGD optimizer
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op

    def get_accuracy_op(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        # correct = tf.nn.in_top_k(logits, labels, 1)
        # return tf.reduce_sum(tf.cast(correct, tf.int32)) / tf.shape(labels)[0]
        return np.NaN

    def loss(self, yhat, y):
        '''
        the l2 norm between yhat and y. In addition, we try to remove the nan value after L2 norm computation.
        yhat: the prediction of the label
        y: the label
        '''
        indiv_loss_no_reg = tf.squared_difference(yhat, y, name='indiv_loss')
        # indiv_loss_no_reg = tf.Print(indiv_loss_no_reg, [yhat[0], y[0], indiv_loss_no_reg[0]])

        # neglect nans when do the average
        loss_no_reg = tf.reduce_mean(tf.boolean_mask(indiv_loss_no_reg,
                                                     tf.logical_not(tf.is_nan(indiv_loss_no_reg))),
                                     name='loss_no_reg')
        # loss_no_reg = tf.Print(loss_no_reg, [loss_no_reg])
        tf.add_to_collection('losses', loss_no_reg)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        # total_loss = tf.Print(total_loss, [yhat[0], y[0], indiv_loss_no_reg[0], loss_no_reg, total_loss])

        return total_loss, loss_no_reg, indiv_loss_no_reg

    def adversarial_loss(self, logits, labels):
        return 0, 0

    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            shp = pl_block.get_shape().as_list()
            shp = [-1 if x is None else x for x in shp]
            feed_dict[pl_block] = np.reshape(vec_block, shp)
        return feed_dict

    def get_inverse_hvp(self, v, approx_type='cg', approx_params=None, verbose=True):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            return self.get_inverse_hvp_lissa(v, **approx_params)
        elif approx_type == 'cg':
            return self.get_inverse_hvp_cg(v, verbose)

    def get_inverse_hvp_lissa(self, v,
                              batch_size=None,
                              scale=10, damping=0.0, num_samples=1, recursion_depth=10000):
        """
        This uses mini-batching; uncomment code for the single sample case.
        """
        inverse_hvp = None
        print_iter = recursion_depth / 10

        for i in range(num_samples):
            # samples = np.random.choice(self.num_train_examples, size=recursion_depth)

            cur_estimate = v

            for j in range(recursion_depth):
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=batch_size)

                feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)
                hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
                cur_estimate = [a + (1 - damping) * b - c / scale
                                for (a, b, c) in zip(v, cur_estimate, hessian_vector_val)]

                # Update: v + (I - Hessian_at_x) * cur_estimate
                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    print(
                        "Recursion at depth %s: norm is %.8lf" % (j, np.linalg.norm(self.list_to_vec((cur_estimate)))))
                    feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)

            if inverse_hvp is None:
                inverse_hvp = [b / scale for b in cur_estimate]
            else:
                inverse_hvp = [a + b / scale for (a, b) in zip(inverse_hvp, cur_estimate)]

        inverse_hvp = [a / num_samples for a in inverse_hvp]
        return inverse_hvp

    def minibatch_hessian_vector_val(self, v):

        num_examples = self.num_train_examples
        if self.mini_batch == True:
            batch_size = 100
            assert num_examples % batch_size == 0
        else:
            batch_size = self.num_train_examples

        num_iter = int(num_examples / batch_size)

        self.reset_datasets()
        hessian_vector_val = None
        for i in range(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=batch_size)
            # Can optimize this
            feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, [v])

            hessian_vector_val_temp = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a, b) in
                                      zip(hessian_vector_val, hessian_vector_val_temp)]

        hessian_vector_val = [a + self.damping * b for (a, b) in zip(hessian_vector_val, v)]

        return hessian_vector_val

    # def minibatch_hessian_vector_val(self, v):
    #     # self.reset_datasets()
    #     feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=self.num_train_examples)
    #     # Can optimize this
    #     feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, [v])
    #     # print(feed_dict)
    #     hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
    #     # hessian_vector_val = self.sess.run(tf.add(hessian_vector_val[0], v[0] * self.damping))
    #     # hessian_vector_val = lambda x, y:  tf.add() elems=(hessian_vector_val[0], v[0])
    #     hessian_vector_val = [a + self.damping * b for (a, b) in zip(hessian_vector_val, v)]
    #     return hessian_vector_val

    def list_to_vec(self, l):
        if not isinstance(l, list) and not isinstance(l, tuple):
            return l.reshape(-1)
        # original = [array([[8.8823157e-08], [4.6933826e-07]], dtype=float32)]
        # after reshape -> [array([8.8823157e-08, 4.6933826e-07], dtype=float32)]
        # after hstakc -> array([8.8823157e-08, 4.6933826e-07], dtype=float32)
        return np.hstack([np.reshape(a, (-1)) for a in l])
        # if not isinstance(l, list) and not isinstance(l, tuple):
        #     return tf.reshape(l, -1)
        # return tf.concat([tf.reshape(a, (-1)) for a in l], axis=1)

    def get_fmin_loss_fn(self, v):
        def get_fmin_loss(x):
            hessian_vector_val = self.list_to_vec(self.minibatch_hessian_vector_val(self.vec_to_list(x)))
            return 0.5 * np.dot(hessian_vector_val, x) - np.dot(self.list_to_vec(v), x)
        return get_fmin_loss

    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
            return self.list_to_vec(hessian_vector_val) - self.list_to_vec(v)
        return get_fmin_grad

    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(p))
        return self.list_to_vec(hessian_vector_val)

    # TODO: still talking about remove a single step, need to be updated to be meaningful
    def get_cg_callback(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)

        def fmin_loss_split(x):
            hessian_vector_val = self.list_to_vec(self.minibatch_hessian_vector_val(self.vec_to_list(x)))
            return 0.5 * np.dot(hessian_vector_val, x), -np.dot(self.list_to_vec(v), x)

        def cg_callback(x):
            # x is current params
            v = self.vec_to_list(x)
            idx_to_remove = 5
            single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
            train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            predicted_loss_diff = np.dot(self.list_to_vec(v),
                                         self.list_to_vec(train_grad_loss_val)) / self.num_train_examples
            if verbose:
                print('Function value: %s' % fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print('Split function value: %s, %s' % (quad, lin))
                print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

        return cg_callback

    def get_inverse_hvp_cg(self, v, verbose):

        fmin_loss_fn = self.get_fmin_loss_fn(v)
        fmin_grad_fn = self.get_fmin_grad_fn(v)
        cg_callback = self.get_cg_callback(v, verbose)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=np.concatenate([np.reshape(x, [-1, 1]) for x in v]),
            fprime=fmin_grad_fn,
            fhess_p=self.get_fmin_hvp,
            callback=cg_callback,
            avextol=1e-8,
            maxiter=100000,
            disp=verbose
        )
        # x0 = np.concatenate([np.reshape(x, [-1, 1]) for x in v])
        # feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        # for pl_block, vec_block in zip(self.v_placeholder, [x0]):
        #     feed_dict[pl_block] = vec_block
        # # return self.vec_to_list(fmin_results)
        # print('hessian_vector_val={} from hvp_cg'.format(self.sess.run(self.hessian_vector, feed_dict=feed_dict)))
        return fmin_results

    def get_inverse_hvp_cg_new(self, v, verbose):
        # self.reset_datasets()
        feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        for pl_block, vec_block in zip(self.v_placeholder, v):
            feed_dict[pl_block] = vec_block
        for ii in range(100):
            self.sess.run(self.training_process, feed_dict=feed_dict)
        return self.sess.run(self.t / 2)

    def get_gradient(self, gradient_op, data_set, indices=None, batch_size=100):
        if indices is None:
            indices = range(len(data_set.labels))

        num_iter = int(np.ceil(len(indices) / batch_size))

        grad_loss_no_reg_val = None
        for i in range(num_iter):
            start = i * batch_size
            end = int(min((i + 1) * batch_size, len(indices)))

            test_feed_dict = self.fill_feed_dict_with_some_ex(data_set, indices[start:end])
            temp = self.sess.run(operate_deep(tf.convert_to_tensor,
                                              gradient_op), feed_dict=test_feed_dict)

            if grad_loss_no_reg_val is None:
                grad_loss_no_reg_val = operate_deep(lambda a: a * (end - start), temp)
            else:
                grad_loss_no_reg_val = \
                    [operate_deep_2v(lambda a, b: a + b * (end - start), x, y)
                     for (x, y) in zip(grad_loss_no_reg_val, temp)]

        grad_loss_no_reg_val = operate_deep(lambda a: a / len(indices), grad_loss_no_reg_val)

        return grad_loss_no_reg_val

    def get_hessian(self, data_set, indices=None, batch_size=100):
        if indices is None:
            indices = range(len(data_set.labels))

        num_iter = int(np.ceil(len(indices) / batch_size))

        grad_loss_no_reg_val = None
        for i in range(num_iter):
            start = i * batch_size
            end = int(min((i + 1) * batch_size, len(indices)))

            test_feed_dict = self.fill_feed_dict_with_some_ex(data_set, indices[start:end])
            temp = self.sess.run(self.hessian, feed_dict=test_feed_dict)

            if grad_loss_no_reg_val is None:
                grad_loss_no_reg_val = operate_deep(lambda a: a * (end - start), temp)
            else:
                grad_loss_no_reg_val = \
                    [operate_deep_2v(lambda a, b: a + b * (end - start), x, y)
                     for (x, y) in zip(grad_loss_no_reg_val, temp)]

        grad_loss_no_reg_val = operate_deep(lambda a: a / len(indices), grad_loss_no_reg_val)

        return grad_loss_no_reg_val

    def get_ich(self, y, data_set, approx_type='cg', approx_params=None, verbose=True):
        # aaa = self.hessian_inverse_vector_product(data_set)
        iter = 300
        init = tf.global_variables_initializer()
        self.sess.run(init)
        data_set = copy.copy(data_set)
        data_set.reset_batch()
        data_set.reference_ts = y
        psi_y = self.sess.run(self.grad_total_loss_op, feed_dict={self.ts_placeholder: y,
                                                                  self.input_index_placeholder: data_set.x,
                                                                  self.labels_index_placeholder: data_set.labels})
        feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        for pl_block, vec_block in zip(self.v_placeholder, psi_y):
            feed_dict[pl_block] = vec_block
        for jj in range(iter):
            self.sess.run(self.training_process, feed_dict=feed_dict)
        ich = self.sess.run(self.t)

        # validations
        # if verbose:
        #    print("To validate the inverse hvp, the following to list should be very close: ")
        #    print(psi_y)
        #    print(self.minibatch_hessian_vector_val(ich))

        return ich

    def __loop_over_gamma(self, series, y_contaminate, gamma, seed, fun):
        n_time_stamp = self.n_time_stamp
        n_non_nan = self.n_non_nan_y_cont
        series = series.copy()
        np.random.seed(seed)
        condition = gamma * n_time_stamp / n_non_nan
        if condition > 1:
            a = np.empty(series.shape)
            a.fill(np.nan)
            res = fun(a)
            return np.insert(res, 0, gamma)
        idx = np.random.binomial(size=n_time_stamp, n=1, p=condition).astype(bool)
        idx = np.logical_and(idx, self.non_nan_y_cont_idx)
        if sum(idx) > 0:
            series[idx, :] = y_contaminate[idx, np.newaxis]
        # return series, gamma
        res = fun(series)
        # print(gamma, seed)
        return np.insert(res, 0, gamma)

    # gamma=0
    def __loop_over_gamma_0(self, series, y_contaminate, gamma, seed, fun):
        n_time_stamp = self.n_time_stamp
        series = series.copy()
        np.random.seed(seed)
        idx = np.random.binomial(size=n_time_stamp, n=1, p=0).astype(bool)
        idx = np.logical_and(idx, self.non_nan_y_cont_idx)
        if sum(idx) > 0:
            series[idx, :] = y_contaminate[idx, np.newaxis]
        res = fun(series)
        # replace res = fun(series) with the original ich function
        # data_set = copy.copy(self.data_sets.train)
        # data_set.reset_batch()
        # data_set.reference_ts = series
        # psi_y = self.sess.run(self.grad_total_loss_op, feed_dict=
        # {self.ts_placeholder: series,
        #  self.input_index_placeholder: data_set.x,
        #  self.labels_index_placeholder: data_set.labels})
        # feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        # for pl_block, vec_block in zip(self.v_placeholder, psi_y):
        #     feed_dict[pl_block] = vec_block
        # iters = 300
        # for ii in range(iters):
        #     self.sess.run(self.training_process, feed_dict=feed_dict)
        # res = self.sess.run(self.t)
        # validations
        # if True:
        #     print("gmma0: To validate the inverse hvp, the following to list should be very close: ")
        #     print(psi_y)
        #     print(self.minibatch_hessian_vector_val(res))
        return np.insert(res, 0, gamma)

    def __expection_over_gamma(self, fun, y_contaminate, gammas, expectation_rep_time, verbose):
        # for gamma = 0, no need to repeat
        if 0 in gammas:
            fun_v_0 = self.__loop_over_gamma_0(self.time_series, y_contaminate, 0, 0, fun)
            fun_v_0 = [fun_v_0] * expectation_rep_time
        else:
            fun_v_0 = []
        large_gamma = self.large_gamma
        rest_gamma = self.rest_gamma
        if len(large_gamma) > 0:
            series = self.time_series.copy()
            idx = self.non_nan_y_cont_idx
            series[idx, :] = y_contaminate[idx, np.newaxis]
            res = fun(series)
            for gamma in large_gamma:
                fun_v_0 += [np.insert(res, 0, gamma)] * expectation_rep_time
        fun_v = []
        # start_time = time.time()
        # series_gamma_pair = []
        for gamma in rest_gamma:
            for i in range(expectation_rep_time):
                fun_v.append(self.__loop_over_gamma(self.time_series, y_contaminate, gamma, i, fun))
                # series_gamma_pair.append((a, b))
        # fun_v = fun(series_gamma_pair)
        # end_time = time.time()
        # print('{} s to compute nested for loop'.format(end_time - start_time))
        df = pd.DataFrame(fun_v_0 + fun_v)
        df_expect_value = df.groupby(df.columns[0]).agg(np.mean)
        slopes = []
        for i in range(df_expect_value.shape[1]):
            slope, _, r_value, _, _ = linregress(df_expect_value.index.values,
                                                 df_expect_value.iloc[:, i].values)
            if verbose:
                print("Regression R-squared: %f" % r_value ** 2)
            elif r_value ** 2 < 0.3:
                print("Regression R-squared: %f" % r_value ** 2)
            slopes.append(slope)
        return np.array(slopes)

    def get_if(self, y_contaminate, gammas=None, expectation_rep_time=10, verbose=True):
        return self.__expection_over_gamma(
            lambda s: self.get_ich(s, self.data_sets.train, verbose=verbose),
            y_contaminate, gammas, expectation_rep_time, verbose)

    def partial_m_partial_gamma(self, index, short_memo, patchy_k, y_contaminate):  # lemma 3.3
        def get_diff(series, index, act_pred):
            return self.sess.run(self.preds, \
                                 self.fill_feed_dict_with_one_ex(self.data_sets.test, index,
                                                                 series))[0, 0] - act_pred[0, 0]

        pred_gamma = 0.0
        act_pred = self.sess.run(self.preds, \
                                 self.fill_feed_dict_with_one_ex(self.data_sets.test, index))
        if short_memo:
            zero_cnt = 0
        max_train = self.data_sets.test.labels[index][0]
        for m in range(max_train - 1, -1, -1):
            series = self.time_series.copy()
            idx = np.arange(m, np.min([m + patchy_k, max_train]))

            replace_by = y_contaminate[idx, np.newaxis]
            nan_idx = np.isnan(replace_by)
            if nan_idx.sum() > 0:  # random sample and get the average
                dif = []
                for i in range(200):
                    replace_by[nan_idx] = \
                        np.random.choice(y_contaminate[np.logical_not(np.isnan(y_contaminate))], nan_idx.sum())
                    series[idx] = replace_by
                    dif.append(get_diff(series, index, act_pred))
                dif = np.mean(dif)
            else:
                series[idx] = replace_by
                dif = get_diff(series, index, act_pred)

            if short_memo:  # not long memory and should be OK to assume far away points has no impact on current values
                if abs(dif) > 1e-5:
                    zero_cnt = 0
                else:
                    zero_cnt += 1
                    if zero_cnt > 10:  # has no contribution for 10 timestamps
                        break

            pred_gamma += dif
        return pred_gamma / patchy_k

    def get_sif(self, y_contaminate, index, patchy_k,
                gammas=None, expectation_rep_time=10,
                verbose=True, is_ar=False, short_memo=True):
        '''
        y_contaminated: contaminating process
        '''
        if gammas is None:
            gammas = np.arange(0.0, 0.09, 0.01)

        # IF value
        init = tf.global_variables_initializer()
        self.sess.run(init)
        start_time = time.time()
        if_v = self.get_if(y_contaminate, gammas, expectation_rep_time, verbose)
        end_time = time.time()
        print('{} s to compute if_v'.format(end_time - start_time))
        start_time = time.time()
        # Pred Over Gamma
        if patchy_k is not None:  # lemma 3.3
            pred_gamma = self.partial_m_partial_gamma(index, short_memo, patchy_k, y_contaminate)
            end_time = time.time()
            print('{} s to compute patchy pred gamma'.format(end_time - start_time))
        elif is_ar:  # lemma 3.4
            pred_gamma = self.sess.run(
                (-tf.reduce_mean(self.labels_placeholder) + np.nanmean(y_contaminate)) * tf.reduce_sum(self.params),
                self.fill_feed_dict_with_all_ex(self.data_sets.train))
            end_time = time.time()
            print('{} s to compute ar pred gamma'.format(end_time - start_time))
        else:
            pred_gamma = self.__expection_over_gamma(
                lambda s: self.sess.run(self.preds,
                                        self.fill_feed_dict_with_one_ex(self.data_sets.test, index, s)),
                y_contaminate,
                gammas, expectation_rep_time,
                verbose)
            pred_gamma = pred_gamma[0]
            end_time = time.time()
            print('{} s to compute third type pred gamma'.format(end_time - start_time))
        start_time = time.time()
        #
        # psi_y = self.sess.run(self.grad_total_loss_op, feed_dict=
        #                 {self.ts_placeholder: y,
        #                  self.input_index_placeholder: data_set.x,
        #                  self.labels_index_placeholder: data_set.labels})
        # Pred Over Theta
        psi_y = self.sess.run(tf.gradients(self.preds, self.params),
                              self.fill_feed_dict_with_one_ex(self.data_sets.test, index))
        end_time = time.time()
        print('{} s to compute psi_y'.format(end_time - start_time))
        res = pred_gamma + np.dot(self.list_to_vec(psi_y), if_v)
        return res

    def find_eigvals_of_hessian(self, num_iter=100, num_prints=10):

        # Setup
        print_iterations = num_iter / num_prints
        feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, 0)

        # Initialize starting vector
        grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=feed_dict)
        initial_v = []

        for a in grad_loss_val:
            initial_v.append(np.random.random(a.shape))
        initial_v, norm_val = normalize_vector(initial_v)

        # Do power iteration to find largest eigenvalue
        print('Starting power iteration to find largest eigenvalue...')

        largest_eig = norm_val
        print('Largest eigenvalue is %s' % largest_eig)

        # Do power iteration to find smallest eigenvalue
        print('Starting power iteration to find smallest eigenvalue...')
        cur_estimate = initial_v
        dotp = -1.0
        for i in range(num_iter):
            cur_estimate, norm_val = normalize_vector(cur_estimate)
            hessian_vector_val = self.minibatch_hessian_vector_val(cur_estimate)
            new_cur_estimate = [a - largest_eig * b for (a, b) in zip(hessian_vector_val, cur_estimate)]

            if i % print_iterations == 0:
                print(-norm_val + largest_eig)
                dotp = np.dot(np.concatenate(new_cur_estimate), np.concatenate(cur_estimate))
                print("dot: %s" % dotp)
            cur_estimate = new_cur_estimate

        smallest_eig = -norm_val + largest_eig
        assert dotp < 0, "Eigenvalue calc failed to find largest eigenvalue"

        print('Largest eigenvalue is %s' % largest_eig)
        print('Smallest eigenvalue is %s' % smallest_eig)
        return largest_eig, smallest_eig

    def update_train_x(self, new_train_x):
        assert np.all(new_train_x.shape == self.data_sets.train.x.shape)
        new_train = DataSet(new_train_x, np.copy(self.data_sets.train.labels))
        self.data_sets = self.Datasets(train=new_train, test=self.data_sets.test)
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        self.reset_datasets()

    def update_train_x_y(self, new_train_x, new_train_y):
        new_train = DataSet(new_train_x, new_train_y)
        self.data_sets = self.Datasets(train=new_train, test=self.data_sets.test)
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        self.num_train_examples = len(new_train_y)
        self.reset_datasets()

    def update_test_x_y(self, new_test_x, new_test_y):
        new_test = DataSet(new_test_x, new_test_y)
        self.data_sets = self.Datasets(train=self.data_sets.train, test=new_test)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)
        self.num_test_examples = len(new_test_y)
        self.reset_datasets()

    @abstractmethod
    def get_all_params(self):
        pass

    @abstractmethod
    def predictions(self, logit):
        pass

    @abstractmethod
    def placeholder_inputs(self):
        pass

    @abstractmethod
    def inference(self, input_x, labels_placeholder=None, keep_probs_placeholder=None):
        pass

