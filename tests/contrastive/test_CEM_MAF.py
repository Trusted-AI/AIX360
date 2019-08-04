import unittest
import tensorflow as tf
import sys, os
import numpy as np
import random

""" This file is a unit test for CEM_MAF. It
1. Loads the model to be explained
2. Loads and predicts on a specific image.
3. Creates an explainer object with two attributes
4. Creates a pertinent positive for the image.

"""

_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_PATH, '../..'))

from aix360.algorithms.contrastive.CEM_MAF import CEM_MAFImageExplainer
from aix360.algorithms.contrastive.CEM_MAF_utils import CELEBAModel
from aix360.algorithms.contrastive.classifiers import KerasClassifier
from aix360.algorithms.contrastive.dwnld_CEM_MAF_celebA import dwnld_CEM_MAF_celebA
from aix360.datasets.celeba_dataset import CelebADataset

class TestCEM_MAFImageExplainer(unittest.TestCase):

    def test_CEM_MAFExplainer(self):

        dwnld = dwnld_CEM_MAF_celebA()

        with tf.Session() as sess: # Open TensorFlow Session
            random.seed(121)
            np.random.seed(1211)
            sess.run(tf.global_variables_initializer())
        
            # Fix attributes to use for test
            attributes = ["Brown_Hair", "High_Cheekbones"]

            # Download pretrained celebA model
            local_path_models = os.path.join(_PATH,'../../aix360/models/CEM_MAF')
            celebA_model_file = dwnld.dwnld_celebA_model(local_path_models)

            # Download attribute functions
            attr_model_files = dwnld.dwnld_celebA_attributes(local_path_models, attributes)

            # Load the pretrained celebA model
            model_file = os.path.join(_PATH,'../../aix360/models/CEM_MAF/celebA')
            loaded_model = CELEBAModel(restore=model_file, use_softmax=False).model
        
            mymodel = KerasClassifier(loaded_model)
        
            # Select an input image, download, load, and process it
            img_ids = [15]
            img_id = img_ids[0]
            local_path_img =  os.path.join(_PATH,'../../aix360/data/celeba_data')
            img_files = dwnld.dwnld_celebA_data(local_path_img, img_ids)
            dataset_obj = CelebADataset(local_path_img) # use the CelebA dataset class
            input_img = dataset_obj.get_img(img_id)
            input_img = np.clip(input_img/2, -0.5, 0.5)
            
            # Predict on input image
            _, orig_class, _ = mymodel.predict_long(input_img)
        
            # Compute classes
            young_flag = orig_class % 2
            smile_flag = (orig_class // 2) % 2
            sex_flag = (orig_class // 4) % 2
            arg_img_name = os.path.join(local_path_img, "{}_img.png".format(img_id))
            print("Image:{}, pred:{}".format(arg_img_name, orig_class))
            print("Male:{}, Smile:{}, Young:{}".format(sex_flag, smile_flag, young_flag))
                
            # designate path to aix360 - needed to find paths to attribute files
            aix360_path = os.path.join(_PATH, '../../aix360') 

            # Create explainer object with two attributes
            attributes = ["Brown_Hair", "High_Cheekbones"]
            explainer = CEM_MAFImageExplainer(mymodel, attributes, aix360_path)
            
            # Set parameters for Pertinent Positive explanation
            arg_mode = 'PP'
            arg_kappa = 5
            arg_gamma = 100.0
            arg_beta = 0.1
            arg_binary_search_steps = 1
            arg_max_iterations = 10
            arg_initial_const = 10
            
            # Run optimizer to find a pertinent positive
            (adv_pp, __, __) = explainer.explain_instance(sess, input_img, None, arg_mode, arg_kappa, 
                                arg_binary_search_steps, arg_max_iterations, 
                                arg_initial_const, arg_gamma, arg_beta)
        
            _, adv_class, _ = mymodel.predict_long(adv_pp)
        
            # Compute class of PP
            young_flag = adv_class % 2
            smile_flag = (adv_class // 2) % 2
            sex_flag = (adv_class // 4) % 2
            print("Pertinent positive pred:{}".format(adv_class))
            print("Male:{}, Smile:{}, Young:{}".format(sex_flag, smile_flag, young_flag))
            
            # remove celebA model file, all attribute model files, and all data files
            os.remove(celebA_model_file[0])
            
            for model_file in attr_model_files:
                os.remove(model_file)
                
            for data_file in img_files:
                os.remove(data_file)
            
if __name__ == '__main__':
    unittest.main()
