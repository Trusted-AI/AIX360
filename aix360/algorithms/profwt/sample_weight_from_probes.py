import numpy as np



def prof_weight_compute(list_probe_filenames,start_layer,last_layer,y_train2):
	"""
	Key function that computes the per sample weights based on probe classifier confidences from various layers. It averages the probe 
	confidences of the layers.

	Parmeters:
	list_probe_filenames (List of strings): A list of strings specifying path where probe confidences for different layers are stored.
	start_layer (int): Index of the starting layer to average with from the list of filenames each storing probe confidences for a layer
	final_layer (int): Index of the last layer to average with from the list of filenames each storing probe confidences for a layer.
	y_train2 (numpy array): Labels corresponding to the set of datasamples in each of the filenames. Dimension - (num of samples x num of classes)
	
	Return:
	prof_weights (numpy array): A vector of weights for each data sample to train the simple model with. Dimension - (num of samples x 1)
	"""
	probe_2_prediction=np.load(list_probe_filenames[start_layer])
	for r in range(start_layer+1,last_layer+1):
		probe_2_prediction=probe_2_prediction+np.load(list_probe_filenames[r])
	##*********************************************
	num_layers_use=last_layer-start_layer+1
	probe_2_prediction=probe_2_prediction.astype(float)/num_layers_use
	num_examples=probe_2_prediction.shape[0]
	prof_weights=np.zeros((num_examples,1))
	##*********
	for r in range(num_examples):
		prof_weights[r]=np.dot(probe_2_prediction[r,:],y_train2[r,:])
	return prof_weights



