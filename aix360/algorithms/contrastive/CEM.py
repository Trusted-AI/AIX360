from __future__ import print_function

from aix360.algorithms.lwbe import LocalWBExplainer

from .CEM_aen import AEADEN

import random
import numpy as np


class CEMExplainer(LocalWBExplainer):
    """
    CEMExplainer can be used to compute contrastive explanations for image and tabular data.
    This is achieved by finding what is minimally sufficient (PP - Pertinent Positive) and
    what should be necessarily absent (PN - Pertinent Negative) to maintain the original classification.
    We use elastic norm regularization to ensure minimality for both parts of the explanation
    i.e. PPs and PNs. An autoencoder can optionally be used to make the explanations more realistic. [#]_

    References:
        .. [#] `Amit Dhurandhar, Pin-Yu Chen, Ronny Luss, Chun-Chen Tu,
           Paishun Ting, Karthikeyan Shanmugam, Payel Das, "Explanations based on
           the Missing: Towards Contrastive Explanations with Pertinent Negatives,"
           Advances in Neural Information Processing Systems (NeurIPS), 2018.
           <https://arxiv.org/abs/1802.07623>`_
    """
    def __init__(self, model):

        """
        Constructor method, initializes the explainer

        Args:
            model: KerasClassifier model whose predictions needs to be explained
        """
        super(CEMExplainer, self).__init__()
        self._wbmodel = model


    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass


    def explain_instance(self, input_X,
                         arg_mode, AE_model, arg_kappa, arg_b,
                         arg_max_iter, arg_init_const, arg_beta, arg_gamma, arg_alpha=0, arg_threshold=1, arg_offset=0):

        """
        Explains an input instance input_X and returns contrastive explanations.
        Note that this assumes that the classifier was trained with inputs normalized in [-arg_offset, arg_offset] range, where arg_offset is 0 or .5.

        Args:
            input_X (numpy.ndarray): input instance to be explained
            arg_mode (str): 'PP' or 'PN'
            AE_model: Auto-encoder model
            arg_kappa (double): Confidence gap between desired class and other classes
            arg_b (double): Number of different weightings of loss function to try
            arg_max_iter (int): For each weighting of loss function number of iterations to search
            arg_init_const (double): Initial weighting of loss function
            arg_beta (double): Weighting of L1 loss
            arg_gamma (double): Weighting of auto-encoder
            arg_alpha (double): Weighting of L2 loss
            arg_threshold (double): automatically turn off all features less than arg_threshold since nothing to turn off
            arg_offset (double): input_X is in [0,1]. we subtract offset when passed to classifier

        Returns:
            tuple:
                * **adv_X** (`numpy ndarray`) -- Perturbed input instance for PP/PN
                * **delta_X** (`numpy ndarray`) -- Difference between input and Perturbed instance
                * **INFO** (`str`) -- Other information about PP/PN
        """

        random.seed(121)
        np.random.seed(1211)

        (_, orig_class, orig_prob_str) = self._wbmodel.predict_long(input_X)
        target_label = orig_class

        target = np.array([np.eye(self._wbmodel._nb_classes)[target_label]])

        # Hard coding batch_size=1
        batch_size = 1

        # Example: for MNIST (1, 28, 28, 1), for tabular (1, no of columns)
        shape = input_X.shape

        attack = AEADEN(self._wbmodel, shape,
                        mode=arg_mode, AE=AE_model, batch_size=batch_size,
                        kappa=arg_kappa, init_learning_rate=1e-2,
                        binary_search_steps=arg_b, max_iterations=arg_max_iter,
                        initial_const=arg_init_const, beta=arg_beta, gamma=arg_gamma,
                        alpha=arg_alpha, threshold=arg_threshold, offset=arg_offset)


        self._wbmodel.predict(input_X) # helps compile
        adv_X = attack.attack(input_X + arg_offset, target)

        adv_prob, adv_class, adv_prob_str = self._wbmodel.predict_long(adv_X)

        delta_X = (input_X + arg_offset) - adv_X - arg_offset # add 0.5 to input for attack but subtract 0.5 to get back to [-0.5, 0.5]

        adv_X = adv_X - arg_offset # subtrack arg_offset to get it back to [-arg_offset, arg_offset]

        _, delta_class, delta_prob_str = self._wbmodel.predict_long(delta_X)

        INFO = "[INFO]kappa:{}, Orig class:{}, Perturbed class:{}, Delta class: {}, Orig prob:{}, Perturbed prob:{}, Delta prob:{}".format(
            arg_kappa, orig_class, adv_class, delta_class, orig_prob_str, adv_prob_str, delta_prob_str)

        return (adv_X, delta_X, INFO)
