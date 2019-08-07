from __future__ import print_function

from aix360.algorithms.lwbe import LocalWBExplainer 
from aix360.algorithms.contrastive.CEM_MAF_aen_PN import AEADEN as AEADEN_PN
from aix360.algorithms.contrastive.CEM_MAF_aen_PP import AEADEN as AEADEN_PP
from tensorflow.contrib.keras.api.keras.models import model_from_json

import os
import sys
import random
import time
import numpy as np
from skimage.segmentation import slic


class CEM_MAFImageExplainer(LocalWBExplainer):
    """CEM_MAFImageExplainer is a Contrastive Image explainer that leverages Monotonic
    Attribute Functions. The main idea here is to explain images using high level semantically meaningful attributes 
    that may either be directly available or learned through supervised or unsupervised methods. [#]_

    References:
        .. [#] `Ronny Luss, Pin-Yu Chen, Amit Dhurandhar, Prasanna Sattigeri,
           Karthikeyan Shanmugam, Chun-Chen Tu, "Generating Contrastive
           Explanations with Monotonic Attribute Functions," 2019.
           <https://arxiv.org/abs/1905.12698>`_
    """
    def __init__(self, model, attributes, aix360_path):
        """Initialize image explainer.

        Currently accepting model input which is an ImageClassifier.
        """
        super(CEM_MAFImageExplainer, self).__init__()
        self._wbmodel = model
        self._attributes = attributes
        self._aix360_path = aix360_path

    def set_params(self, *argv, **kwargs):
        """Set parameters for the explainer."""
        pass

    def explain_instance(self, sess, input_img, input_latent, arg_mode, arg_kappa, arg_binary_search_steps,
                        arg_max_iterations, arg_initial_const, arg_gamma, arg_beta, arg_attr_reg=1,
                        arg_attr_penalty_reg=1, arg_latent_square_loss_reg=1):
        """Explains an input instance input_image e.g. celebA is shape (1, 224, 224, 3)

        Hard coded batch_size=1, assuming we provide explanation for 1 input_image at a time. Returns
        either pertinent positive or pertinent depending on parameter.

        Args:
            sess (tensorflow.python.client.session.Session): Tensorflow session
            input_img (numpy.ndarray): image to be explained, of shape (1, size, size, channels)
            input_latent (numpy.ndarray): image to be explained, of shape (1, size, size, channels)
                in the latent space
            arg_mode (str): "PN" for pertinent negative or "PP" for pertinent positive
            arg_kappa (float): Confidence parameter that controls difference between prediction of
                PN (or PP) and original prediction
            arg_binary_search_steps (int): Controls number of random restarts to find best PN or PP
            arg_max_iterations (int): Max number iterations to run some version of gradient descent on
                PN or PP optimization problem from a single random initialization, i.e., total
                number of iterations wll be arg_binary_search_steps * arg_max_iterations
            arg_initial_const (int): Constant used for upper/lower bounds in binary search
            arg_gamma (float): Penalty parameter encouraging addition of attributes for PN or PP
            arg_beta (float): Penalty parameter encourages minimal addition of attributes to PN
                or sparsity of the mask that generates the PP
            arg_attr_reg (float): Penalty parameter on regularization of PN to be predicted different from
                original image
            arg_attr_penalty_reg (float): Penalty regularizing PN from being too different from original image
            arg_latent_square_loss_reg (float): Penalty regularizing PN from being too different from original
                image in the latent space

        Returns:
            tuple:
                * **adv_img** (`numpy.ndarray`) -- the pertinent positive or the pertinent negative image
                * **attr_mod** (`str`) -- only for PN; a string detailing which attributes were modified from the
                  original image
                * **INFO** (`str`) -- only for PN; a string of information about original vs PN class and
                  original vs PN prediction probability
        """

        random.seed(121)
        np.random.seed(1211)

        # %%change%%
        #(orig_prob, orig_class, orig_prob_str) = util.model_prediction(model, input_img)
        (orig_prob, orig_class, orig_prob_str) = self._wbmodel.predict_long(input_img)

        if arg_mode == 'PN':
            target_label = [np.eye(self._wbmodel._nb_classes)[orig_class]]


            attack_pn = AEADEN_PN(sess, self._wbmodel, attributes=self._attributes, aix360_path=self._aix360_path,
                            mode = arg_mode, batch_size=1, kappa=arg_kappa, init_learning_rate=1e-2,
                            binary_search_steps=arg_binary_search_steps, max_iterations=arg_max_iterations,
                            initial_const=arg_initial_const, gamma=arg_gamma, attr_reg=arg_attr_reg,
                            attr_penalty_reg=arg_attr_penalty_reg, latent_square_loss_reg=arg_latent_square_loss_reg)

            adv_img = attack_pn.attack(input_img, target_label, input_latent)
            adv_prob, adv_class, adv_prob_str = self._wbmodel.predict_long(adv_img)
            attr_mod = self.check_attributes_celebA(self._attributes, input_img, adv_img)
            
            INFO = "[INFO] Orig class:{}, Adv class:{}, Orig prob:{}, Adv prob:{}".format(orig_class, adv_class, orig_prob_str, adv_prob_str)

        else: # assume arg_mode is PP
            print("Creating a mask for pertinent positive")
            # create mask
            arg_seg_number = 200
            # Segment the original image using and create a mask for the segmentation
            #data = CELEBA_wrapper(os.path.join(img_path, "{}_img.npy".format(img_id)), orig_class, model)
            mask_label = slic(input_img, n_segments=arg_seg_number)[0]
            mask_num = len(np.unique(mask_label))
            mask_size = mask_label.shape[0]
            mask_mat = np.zeros((mask_num, mask_size, mask_size))
            for i in range(mask_num):
                temp_idx = np.argwhere(mask_label==i)
                for j in temp_idx:
                    mask_mat[(i,) + tuple(j)] = 1

            attack_pp = AEADEN_PP(sess, self._wbmodel, mask_mat=mask_mat, mode=arg_mode, batch_size=1, \
                                kappa=arg_kappa, init_learning_rate=1e-2, binary_search_steps=arg_binary_search_steps, \
                                max_iterations=arg_max_iterations, initial_const=arg_initial_const, beta=arg_beta, \
                                gamma=arg_gamma, attributes=self._attributes, aix360_path=self._aix360_path)


            target = np.zeros(self._wbmodel._nb_classes)
            target[orig_class]=1
            adv_img, img_mask = attack_pp.attack(input_img, [target])
            adv_prob, adv_class, adv_prob_str = self._wbmodel.predict_long(adv_img)

            print('Generating the pertinent positive')
            # Generate the PP
            success = False
            print("Start ranking:")
            mask_vec = img_mask.reshape(-1)
            sort_idx = np.argsort(mask_vec)
            total_nonezero = len(np.argsort(mask_vec>0))
            working_mask = np.zeros((1,) + (mask_size, mask_size) + (1,))
            for i in range(1,total_nonezero):
                temp_index = sort_idx[-i]
                mask_position = np.argwhere(mask_mat[temp_index]==1)
                for index in mask_position:
                    working_mask[(0,) + tuple(index) + (0,)] = 1
                adv_img = working_mask * input_img
                img_prob, img_class, img_prob_str = self._wbmodel.predict_long(adv_img)
                print("i:{}, index:{}, value:{}, class:{}".format(i, temp_index, mask_vec[temp_index], img_class))
                if img_class == orig_class:
                    success = True
                    break

            attr_mod = None
            INFO = None

        return(adv_img, attr_mod, INFO)

      
    def check_attributes_celebA(self, attributes, x, y): 
        """
        Load attribute classifiers and check which attributes in original image x
        are modified in adversarial image y

        Args:
            attributes (str list): list of attributes to load attribute classifiers for
            x (numpy.ndarray): original image
            y (numpy.ndarray): adversarial image

        Returns:
            str: string detailing which attributes were added to (or removed from)
            x resulting in y
        """

        orig_attr_score = np.zeros((len(attributes),1))
        adv_attr_score = np.zeros((len(attributes),1))
        for i in range(len(attributes)):
            attr = attributes[i]
            # load json and create model
            json_file_name = "../../aix360/models/CEM_MAF/simple_{}_model.json".format(attr)
            json_file = open(json_file_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            weight_file_name = "../../aix360/models/CEM_MAF/simple_{}_weights.h5".format(attr)
            loaded_model.load_weights(weight_file_name)

            orig_attr_score[i] = loaded_model.predict(x)[0]
            adv_attr_score[i] = loaded_model.predict(y)[0]

        # pre-determined thresholds for changes in prediction values
        thresh_pos = np.zeros((len(attributes),1))
        thresh_pos[0] = .15
        thresh_pos[1] = .15
        thresh_pos[2] = .15
        thresh_pos[3] = .15
        thresh_pos[4] = .15
        thresh_pos[5] = .15
        thresh_pos[6] = .1
        thresh_pos[7] = .25
        thresh_pos[8] = .1
        thresh_pos[9] = .15
        thresh_pos[10] = .15
        thresh_pos[11] = .15

        thresh_neg = np.zeros((len(attributes),1))
        thresh_neg[0] = -.25
        thresh_neg[1] = -.25
        thresh_neg[2] = -.25
        thresh_neg[3] = -.25
        thresh_neg[4] = -.35
        thresh_neg[5] = -.25
        thresh_neg[6] = -.12
        thresh_neg[7] = -.25
        thresh_neg[8] = -.25
        thresh_neg[9] = -.25
        thresh_neg[10] = -.25
        thresh_neg[11] = -.25

        changes_abs = adv_attr_score - orig_attr_score
        changes = np.zeros((len(attributes),1))
        res = ""
        for i in range(len(attributes)):
            if changes_abs[i] >= thresh_pos[i]:
                changes[i] = 1
            elif changes_abs[i] <= thresh_neg[i]:
                changes[i] = -1
        added = np.where(changes == 1)[0]
        for j in range(len(added)):
            res += "Added " + attributes[added[j]] + ","
        removed = np.where(changes[i] == -1)[0]
        for j in range(len(removed)):
            res += "Removed " + attributes[removed[j]] + ","
        return res[:-1]
