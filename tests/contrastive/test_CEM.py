import unittest
import os
import shutil

from aix360.algorithms.contrastive import CEMExplainer
from aix360.algorithms.contrastive import KerasClassifier
from aix360.datasets import MNISTDataset
from keras.models import model_from_json

import matplotlib.pyplot as plt
import numpy as np


class TestCEMExplainer(unittest.TestCase):

    def load_model(self, model_json_file, model_wt_file):

        # read model json file
        with open(model_json_file, 'r') as f:
            model = model_from_json(f.read())

        # read model weights file
        model.load_weights(model_wt_file)

        return model


    def test_CEM(self):

        # set up

        _PATH = os.path.dirname(os.path.realpath(__file__))

        fp = os.path.join(_PATH, "cem_tests_results")
        if not os.path.exists(fp):
            os.makedirs(fp)

        data = MNISTDataset()

        ae_js = os.path.join(_PATH, "../../aix360/models/CEM/mnist_AE_1_decoder.json")
        ae_wt = os.path.join(_PATH, "../../aix360/models/CEM/mnist_AE_1_decoder.h5")

        ae_model = self.load_model(ae_js, ae_wt)

        model_js = os.path.join(_PATH, "../../aix360/models/CEM/mnist.json")
        model_wt = os.path.join(_PATH, "../../aix360/models/CEM/mnist")

        mnist_model = self.load_model(model_js, model_wt)

        mymodel = KerasClassifier(mnist_model)

        explainer = CEMExplainer(mymodel)

        # Explain an input instance
        image_id = 340
        input_image = data.test_data[image_id]
        plt.imshow(input_image[:,:,0], cmap="gray")
        plt.savefig(os.path.join(fp, 'input_image_340_d3.png'))

        # check prediction of model is "3" as expected
        self.assertEqual(mymodel.predict_classes(np.expand_dims(input_image, axis=0)), 3)

        # Obtain Pertinent Negative explanation
        arg_max_iter = 1000
        arg_b = 9
        arg_init_const = 10.0
        arg_mode = "PN"
        arg_kappa = 10
        arg_beta = 1e-1
        arg_gamma = 100

        (adv_pn, delta_pn, _) = explainer.explain_instance(np.expand_dims(input_image, axis=0),
                                                           arg_mode, ae_model, arg_kappa, arg_b,
                                                           arg_max_iter, arg_init_const, arg_beta, arg_gamma)

        # Obtain Pertinent Positive explanation
        arg_mode = "PP"

        (adv_pp, delta_pp, _) = explainer.explain_instance(np.expand_dims(input_image, axis=0),
                                                           arg_mode, ae_model, arg_kappa, arg_b,
                                                           arg_max_iter, arg_init_const, arg_beta, arg_gamma)

        fig0 = (input_image[:, :, 0] + 0.5) * 255
        fig1 = (adv_pn[0, :, :, 0] + 0.5) * 255
        fig2 = (fig1 - fig0)  # rescaled delta_pn
        fig3 = (adv_pp[0, :, :, 0] + 0.5) * 255
        fig4 = (fig0 - fig3)  # rescaled delta_pp

        _, axarr = plt.subplots(1, 5)
        axarr[0].set_title("Orig" + "(" + str(mymodel.predict_classes(np.expand_dims(input_image, axis=0))[0]) + ")")
        axarr[1].set_title("Orig + PN" + "(" + str(mymodel.predict_classes(adv_pn)[0]) + ")")
        axarr[2].set_title("PN")
        axarr[3].set_title("Orig + PP")
        axarr[4].set_title("PP" + "(" + str(mymodel.predict_classes(delta_pp)[0]) + ")")

        axarr[0].imshow(fig0, cmap="gray")
        axarr[1].imshow(fig1, cmap="gray")
        axarr[2].imshow(fig2, cmap="gray")
        axarr[3].imshow(fig3, cmap="gray")
        axarr[4].imshow(fig4, cmap="gray")
        plt.savefig(os.path.join(fp, 'predict_classes_4.png'))

        if os.path.exists(fp):
            shutil.rmtree(fp)

if __name__ == '__main__':
    unittest.main()