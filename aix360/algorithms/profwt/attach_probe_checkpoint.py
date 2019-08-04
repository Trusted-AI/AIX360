import tensorflow as tf
import aix360.algorithms.profwt.resnet_keras_model
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import json



def print_layer_labels(checkpoint_path):
	"""
	Loads the tensorflow computation graph from the checkpoint
	Prints all operations names defined in the computation graph defined by the checkpoint.

	Parameters:
	checkpoint_path (string): Path to the checkpoint filename where the tensorflow checkpoint is stored.
	"""
	g= tf.train.import_meta_graph(checkpoint_path+'.meta')
	graph = tf.get_default_graph()
	print(graph.get_operations())
	tf.reset_default_graph()
	return

def attach_probe_eval(input_features_name,label_name,operation_name,x_train2,y_train2,checkpoint_path):
	"""
	Flattens a layer specificed by operation name and then evaluates the flattened layer on a dataset.
	
	Parameters:
	input_features_name (string): String that specifies the feature input placeholder tensor
	input_label_name (string): String that specifies the label placeholder tensor
	operation_name (string): String that specifies the operation at the end of which the layer outputs need to be probed. Has to be one of the values in the list
	                printed by the print_layer_labels function
	x_train2 (numpy array): training samples on which the the flattened layers needs to be evalauted on.
	y_train2 (numpy array): labels of training samples on which the flattened layer needs to be evaulated on. The function does not use this 
				information but is needed to supply values to the label placeholder in a keras/tensorflow model.

	Return:
	probe_tensor_val (numpy array)- Evaluated layer output matrix of dimensions (num of samples x flattened layer dimension).

	"""

	# Build the computation graph correpsonding to the trained model from the checkpoint
	g= tf.train.import_meta_graph(checkpoint_path+'.meta')
	graph = tf.get_default_graph()
	# Get the tensor corresponding to the operation
	a=graph.get_tensor_by_name(operation_name)
	#Get the placeholders for inputs and labels.
	input_x=graph.get_tensor_by_name(input_features_name)
	label_y=graph.get_tensor_by_name(label_name)
	print(a)
	# Stop any gradient flowing through it
	a_sg = tf.stop_gradient(a)
	#Flatten and obtain a probe output.
	shape = a_sg.get_shape().as_list()     # a list: [None, 9, 2]
	dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
	probe = tf.reshape(a_sg, [-1, dim])
	print(probe)

	# Loads the weights in the computation graph and evaluate the probe tensor on all samples provided
	batch_size=500

	saver=tf.train.Saver()
	with tf.Session() as sess:      
		saver.restore(sess,checkpoint_path)
		num_examples=len(x_train2)
		i=0
		for offset in range(0, num_examples, batch_size):
			print (i)
			end = offset + batch_size
			x_t,y_t=x_train2[offset:end],y_train2[offset:end]
			if i==0:
				probe_list=sess.run(probe,feed_dict={input_x: x_t, label_y: y_t})
			else:
				l_dum=sess.run(probe,feed_dict={input_x: x_t, label_y: y_t})
				probe_list=np.append(np.array(probe_list),np.array(l_dum),axis=0)  
			i=i+1
	tf.reset_default_graph()
	#Save the array of flattend probe outputs
	probe_tensor_val=probe_list
	assert str(type(probe_tensor_val))=="<class 'numpy.ndarray'>"  
	assert probe_tensor_val.shape[0]==len(x_train2)
	print('probe shape',probe_tensor_val.shape)
	return probe_tensor_val



