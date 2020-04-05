

import time
import six
import sys

#import cifar_input
import numpy as np
import aix360.algorithms.profwt.resnet_keras_model
import tensorflow as tf
from sklearn.utils import shuffle
import json


dataset='cifar10' # 'cifar10 or cifar100.'



def fully_connected(x, out_dim):
  """FullyConnected layer
  
  Parameters:
  x (Tensor): Input tensor to the fully connected layer
  out_dim (int): Output dimension of the fully connected layer.
  
  Return: 
  The Tensor corresponding to the fully connected layer output.
  """
  w = tf.get_variable(
      'DW', [x.get_shape()[1], out_dim],
      initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
  b = tf.get_variable('biases', [out_dim],
                      initializer=tf.constant_initializer())
  return tf.nn.xw_plus_b(x, w, b)

def probe_train_eval(probe_train_input,y_train1,num_classes,probe_eval_input,y_train2,to_save_probe_model_filename):
  """ 
  Given the input corresponding to a previously flattened layer output stored in a file, this trains a simple logistic model
  with the flattened layer output at the input to predict the label. We call this logistic model a probe classifier. Probe 
  classifier is then evaulated on a set of layer outputs of a different dataset.
  
  Parameters:
  # probe_train_input (numpy array): Layer output treated as the input to train the probe classifier on. Dim - (num of samples_1 x flattened layer output dimension)
  # y_train1 (numpy array): Labels corresponding to probe_train_input that is used to train the probe classifier. (num of samples_1 x num of classes)
  # probe_eval_input (numpy array): Layer output belonging to (perhaps) a different dataset to evaluate probe classifier on. Dimension - (num of samples_2 x flattened layer output dimension)
  # y_train2 (numpy array): Labels corresponding to probe_eval_input that is used to evaluate the probe classifier. Dimension - (num of samples_2 x num of classes)
  # to_save_probe_model_filename (string): Filename to save the probe classifier trained on probe_eval_input.

  Return:
  array_logits (numpy array): Matrix of logits of the Probe Classifier of dimensions (number of samples x num of classes)
  array_pred (numpy array): Matrix of probabilities of the Probe Classifier of dimensions (number of samples x num of classes)
  """

  probe_val=probe_train_input

  # Tensorflow Graph Definition
  columns=probe_val.shape[1]
  probe_optimizer='mom'

  probe_input=tf.placeholder(tf.float32,shape=(None,columns))
  probe_labels=tf.placeholder(tf.float32,shape=(None,num_classes))

  probe_first_layer=probe_input

  with tf.variable_scope('probe_logits'):
    probe_logits=fully_connected(probe_first_layer,num_classes)
    probe_predictions = tf.nn.softmax(probe_logits)
    probe_cost_1 = tf.nn.softmax_cross_entropy_with_logits(
          logits=probe_logits, labels=probe_labels)
    probe_cost_2 = tf.reduce_mean(probe_cost_1, name='xent')

  # Tensorflow training operations defined.
  learning_rate=0.1
  trainable_variables = tf.trainable_variables()
  grads = tf.gradients(probe_cost_2,trainable_variables)

  if probe_optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif probe_optimizer == 'mom':
    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9)
  probe_train_op = optimizer.apply_gradients(zip(grads, trainable_variables),name='train_step')

  probe_truth = tf.argmax(probe_labels, axis=1)
  predictions = tf.argmax(probe_predictions, axis=1)
  probe_precision = tf.reduce_mean(tf.to_float(tf.equal(predictions,probe_truth)))
  epochs=200
  num_examples=len(probe_val)
  batch_size=128

  saver=tf.train.Saver()
  probe_val_2=probe_eval_input

  print("Start Training Probe Model.....")
  # Training Session that runs the training operation after initialization
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    global_step=0
    for i in range(epochs):
      print(i)
      x_t, y_t = shuffle(probe_val,y_train1)
      for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        batch_x, batch_y = x_t[offset:end,:], y_t[offset:end,:]
        sess.run(probe_train_op,feed_dict={probe_input: batch_x, probe_labels: batch_y})
        global_step=global_step+1

      if (i+1)%100==0:
        saver.save(sess,to_save_probe_model_filename)
    print("Starting to Evaluate Probe Model Confidences and Saving in a File.....")
    saver.restore(sess,to_save_probe_model_filename)
    batch_size=500
    num_examples=len(probe_val_2)
    array_logits=[]
    array_pred=[]
    precision=0
    l=0
    for offset in range(0, num_examples, batch_size):
      print (l)
      end = offset + batch_size
      batch_x, batch_y = probe_val_2[offset:end,:], y_train2[offset:end,:]
      if l==0:
        array_logits=sess.run(probe_logits,feed_dict={probe_input: batch_x, probe_labels : batch_y})
        array_pred=sess.run(probe_predictions,feed_dict={probe_input: batch_x, probe_labels : batch_y})
      else:
        arr_dummy_logits=sess.run(probe_logits,feed_dict={probe_input: batch_x, probe_labels: batch_y})
        arr_dummy_pred=sess.run(probe_predictions,feed_dict={probe_input: batch_x, probe_labels: batch_y})
        array_logits=np.append(np.array(array_logits),np.array(arr_dummy_logits),axis=0) 
        array_pred=np.append(np.array(array_pred),np.array(arr_dummy_pred),axis=0)
      l=l+1

  print(array_logits.shape)
  print(array_pred.shape)

  print("Probe Confidences/Logits Saved...")
  tf.reset_default_graph()
  return (array_logits,array_pred)





