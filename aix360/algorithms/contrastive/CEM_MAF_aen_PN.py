## aen_attack.py -- attack a network optimizing elastic-net distance with an en decision rule
##                  when autoencoder loss is applied
##
## Copyright (C) 2018, PaiShun Ting <paishun@umich.edu>
##                     Chun-Chen Tu <timtu@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
## Copyright (C) 2017, Yash Sharma <ysharma1126@gmail.com>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the "supplementary license" folder present in the root directory.
##
## Modifications Copyright (c) 2019 IBM Corporation


import sys
import hashlib
import json
import tensorflow as tf
import numpy as np
from safetensors.numpy import load_file as load_safetensors
from tensorflow.contrib.keras.api.keras.models import Model, Sequential, model_from_json
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
import os
from contextlib import nullcontext as _nullcontext

from aix360.algorithms.contrastive.gan_networks import G_paper

# SHA-256 of the trusted GAN artifacts under aix360/models/CEM_MAF/gan/.
# Pinned to detect tampering or accidental corruption regardless of how the
# file reached disk (downloader, manual copy, etc.).
_EXPECTED_SHA256 = {
    "gan_weights.safetensors": "3897bd712f7db7619c6e32eee8955e2d6714fcb4a3fdbc7f56267c2c69129d19",
    "gan_static_kwargs.json":  "9215b500feeb4bcaa5d388d9f8087a07be3eb065d8321248ce8f4ac5757941d6",
}


def _verify_sha256(path):
    name = os.path.basename(path)
    if name not in _EXPECTED_SHA256:
        raise RuntimeError("No pinned hash for {}".format(name))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    expected = _EXPECTED_SHA256[name]
    if actual != expected:
        raise RuntimeError(
            "SHA-256 mismatch for {}: expected {}, got {}".format(path, expected, actual)
        )

class AEADEN:
    def __init__(self, sess, model, attributes, aix360_path, mode, batch_size, kappa, init_learning_rate, binary_search_steps, max_iterations, initial_const, gamma, attr_reg, attr_penalty_reg, latent_square_loss_reg, gan_device=None, attr_classifier_device=None):
        """
        Initialize PN explainer object. 
        
        Args:
            sess (tensorflow.python.client.session.Session): Tensorflow session
            model: KerasClassifier that contains a trained model to be explained
            attributes (str list): list of attributes to load attribute classifiers for
            aix360_path (str): path to aix360 used to determine paths to pretrained attribute classifiers  
            mode (str): "PN" for pertinent negative or "PP" for pertinent positive
            batch_size (int): batch size for how many instances to explain
            kappa (float): Confidence parameter that controls difference between prediction of
                PN (or PP) and original prediction
            init_learning_rate (float): initial learning rate for gradient descent optimizer
            binary_search_steps (int): Controls number of random restarts to find best PN
            max_iterations (int): Max number iterations to run some version of gradient descent on
                PN optimization problem from a single random initialization, i.e., total 
                number of iterations wll be arg_binary_search_steps * arg_max_iterations
            initial_const (int): Constant used for upper/lower bounds in binary search
            gamma (float): Penalty parameter encouraging addition of attributes for PN
            attr_reg (float): Penalty parameter on regularization of PN to be predicted different from 
                original image
            attr_penalty_reg (float): Penalty regularizing PN from being too different from original image
            latent_square_loss_reg (float): Penalty regularizing PN from being too different from original
                image in the latent space
            gan_device (str or None): TF device for the GAN forward pass.
                Pass '/cpu:0' on Ampere/Hopper (A100/H100) — the bundled
                cuBLAS-10 in TF1.15 has no SGEMM kernels for sm_80 and
                NaN-corrupts the GAN forward through pixel_norm's rsqrt.
            attr_classifier_device (str or None): TF device for the attribute
                classifiers. Same TF1.15 + cuBLAS-10 + Ampere SGEMM bug — pass
                '/cpu:0' on A100/H100 or attribute classifier outputs are
                non-deterministic / NaN-corrupted, which silently corrupts
                the attr_score and attr_penalty terms in the loss.
        """


#        image_size, num_channels, nun_classes = model.image_size, model.num_channels, model.num_labels
        # %%change%%
        image_size = model._input_shape[0]
        num_channels = model._input_shape[2]
        nun_classes = model._nb_classes 
        shape = (batch_size, image_size, image_size, num_channels)
        latent_shape = (batch_size, 512)

        self.sess = sess
        self.INIT_LEARNING_RATE = init_learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.kappa = kappa
        self.init_const = initial_const
        self.batch_size = batch_size
        self.mode = mode
        self.gamma = gamma
        self.attributes = attributes
        self.aix360_path = aix360_path
        self.attr_reg = attr_reg
        self.attr_penalty_reg = attr_penalty_reg
        self.attr_threshold = tf.constant(0.5, dtype="float32") # penalize the attributes of orig_img having scores <= this value
        self.latent_square_loss_reg = latent_square_loss_reg

        # these are variables to be more efficient in sending data to tf
        self.orig_img = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.orig_latent = tf.Variable(np.zeros(latent_shape), dtype=tf.float32, name="orig_latent")
        self.adv_latent = tf.Variable(np.zeros(latent_shape), dtype=tf.float32, name="adv_latent")
        self.target_lab = tf.Variable(np.zeros((batch_size,nun_classes)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.global_step = tf.Variable(0.0, trainable=False)

        # and here's what we use to assign them
        self.assign_orig_img = tf.placeholder(tf.float32, shape)
        self.assign_orig_latent = tf.placeholder(tf.float32, latent_shape, name="assign_orig_latent")
        self.assign_adv_latent = tf.placeholder(tf.float32, latent_shape, name="assign_adv_latent")
        self.assign_target_lab = tf.placeholder(tf.float32, (batch_size,nun_classes), name="assign_target_label")
        self.assign_const = tf.placeholder(tf.float32, [batch_size])

        ### Load attribute classifier
        nn_type = "simple"
        #import copy
        attr_model_list=[]
        # Optional device pin for the attribute classifiers. Same TF1.15 +
        # cuBLAS-10 + Ampere SGEMM bug as gan_device — pass
        # attr_classifier_device='/cpu:0' on A100/H100 or these conv
        # classifiers return non-deterministic / NaN-corrupted predictions,
        # which silently corrupts attr_score / attr_penalty in the loss.
        self.attr_classifier_device = attr_classifier_device
        attr_device_ctx = tf.device(attr_classifier_device) if attr_classifier_device else _nullcontext()
        with attr_device_ctx:
            for attr in self.attributes:
                # load json and create model
                json_file_name = os.path.join(aix360_path, "models/CEM_MAF/attr_model/{}_{}_model.json".format(nn_type, attr))
                json_file = open(json_file_name, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                # load weights into new model
                weight_file_name = os.path.join(aix360_path, "models/CEM_MAF/attr_model/{}_{}_weights.h5".format(nn_type, attr))
                loaded_model.load_weights(weight_file_name)
                print("Loaded model for " + attr + " from disk")
                attr_model_list.append(loaded_model)
        print("# of attr models is",len(attr_model_list))
#        print("# of attr smaller than THR is",len(attr_threshold_idx))

        # Load GAN from hash-verified safetensors + JSON sidecar.
        # Replaces a pickle.load that exec()'d embedded source (CWE-502).
        gan_dir = os.path.join(aix360_path, "models", "CEM_MAF", "gan")
        weights_path = os.path.join(gan_dir, "gan_weights.safetensors")
        kwargs_path = os.path.join(gan_dir, "gan_static_kwargs.json")
        _verify_sha256(weights_path)
        _verify_sha256(kwargs_path)
        with open(kwargs_path) as f:
            gan_meta = json.load(f)
        gan_weights = load_safetensors(weights_path)

        in_labels = tf.constant(0, shape=(1, 0))
        gan_scope = gan_meta["scope"]
        # Optional device pin for the GAN. TF1.15's bundled cuBLAS-10 has no
        # SGEMM kernels for Ampere (sm_80, e.g. A100), which silently
        # corrupts the GAN forward to NaN/Inf via pixel_norm's rsqrt — pass
        # gan_device='/cpu:0' on Ampere/Hopper. On pre-Ampere GPUs (V100,
        # T4, P100, ...) leave gan_device=None for full-GPU speed.
        gan_device_ctx = tf.device(gan_device) if gan_device else _nullcontext()
        with gan_device_ctx, tf.variable_scope(gan_scope, reuse=tf.AUTO_REUSE):
            # G_paper now emits NHWC directly (networks.py was patched to
            # NHWC so backward works on CPU and so non-NCHW kernels can
            # dispatch on Ampere GPUs). The earlier explicit transpose is
            # therefore unnecessary.
            out_image = G_paper(self.adv_latent, in_labels, **gan_meta["static_kwargs"])
            resize_image = tf.image.resize_images(out_image, [224, 224])
            self.adv_img = tf.clip_by_value(resize_image / 2, -0.5, 0.5)

        gan_assigns = []
        for var in tf.global_variables():
            if not var.name.startswith(gan_scope + "/"):
                continue
            localname = var.name[len(gan_scope) + 1:].split(":")[0]
            if localname in gan_weights:
                gan_assigns.append(tf.assign(var, gan_weights[localname]))
        if not gan_assigns:
            raise RuntimeError("No GAN variables matched the safetensors keys under scope {}".format(gan_scope))
        self.gan_assigns = gan_assigns
        self._gan_weights_loaded = False
        
        self.adv_updater = tf.assign(self.adv_latent, self.assign_adv_latent)

        """--------------------------------"""
        # prediction BEFORE-SOFTMAX of the model
        self.delta_img = self.orig_img - self.adv_img
        # %%change%%
        if self.mode == "PP":
#            self.ImgToEnforceLabel_Score = model.predict(self.delta_img)
            self.ImgToEnforceLabel_Score = model.predictsym(self.delta_img)
        elif self.mode == "PN":
#            self.ImgToEnforceLabel_Score = model.predict(self.adv_img)
            self.ImgToEnforceLabel_Score = model.predictsym(self.adv_img)
        # Attribute classifier score
        self.attr_score = tf.constant(0, dtype="float32")
        self.attr_penalty = tf.constant(0, dtype="float32")
        # Re-enter the attr-classifier device context for the symbolic forward
        # passes — without this, Conv2D ops can be placed on GPU even though
        # variables live on CPU, which still hits the cuBLAS-10 SGEMM bug.
        attr_use_ctx = tf.device(self.attr_classifier_device) if self.attr_classifier_device else _nullcontext()
        with attr_use_ctx:
            if self.mode == "PP":
                for i in range(len(attr_model_list)):
                    self.attr_score = self.attr_score + tf.maximum(attr_model_list[i](self.adv_img) - attr_model_list[i](self.orig_img),tf.constant(0, tf.float32))
                    self.attr_score = tf.squeeze(self.attr_score)
            elif self.mode == "PN":
                for i in range(len(attr_model_list)):
                    self.attr_score = self.attr_score + tf.maximum(attr_model_list[i](self.orig_img) - attr_model_list[i](self.adv_img),tf.constant(0, tf.float32))
                    self.attr_score = tf.squeeze(self.attr_score)
                    self.attr_penalty = self.attr_penalty + tf.multiply(tf.cond(tf.squeeze(attr_model_list[i](self.orig_img)) <= self.attr_threshold, lambda: tf.constant(1, tf.float32), lambda: tf.constant(0, tf.float32)),tf.squeeze(attr_model_list[i](self.adv_img)))
        # Sum of attributes penalty in attr_threshold_idx  
 #       self.attr_penalty = tf.constant(0, dtype="float32")
 #       if len(attr_threshold_idx)==0:
 #           pass
 #       else:  
 #           for i in range(len(attr_threshold_idx)):
 #                self.attr_penalty =  self.attr_penalty + attr_model_list[i](self.adv_img)

        self.delta_latent = self.orig_latent - self.adv_latent

        # distance to the input data
        self.L2_img_dist = tf.reduce_sum(tf.square(self.delta_img),[1,2,3])
        self.L2_latent_dist = tf.reduce_sum(tf.square(self.delta_latent))



        # compute the probability of the label class versus the maximum other
        self.target_lab_score        = tf.reduce_sum((self.target_lab)*self.ImgToEnforceLabel_Score,1)
        self.max_nontarget_lab_score = tf.reduce_max((1-self.target_lab)*self.ImgToEnforceLabel_Score - (self.target_lab*10000),1)
        if self.mode == "PP":
            Loss_Attack = tf.maximum(0.0, self.max_nontarget_lab_score - self.target_lab_score + self.kappa)
        elif self.mode == "PN":
            Loss_Attack = tf.maximum(0.0, -self.max_nontarget_lab_score + self.target_lab_score + self.kappa)
        # sum up the losses
        self.Loss_Latent_L2Dist    = tf.reduce_sum(self.latent_square_loss_reg*self.L2_latent_dist)
        self.Loss_Img_L2Dist    = tf.reduce_sum(self.gamma*self.L2_img_dist)
        self.Loss_Attack    = tf.reduce_sum(self.const*Loss_Attack)
        self.Loss_attr = tf.reduce_sum(self.attr_reg*self.attr_score)
        self.Loss_attr_penalty = tf.reduce_sum(self.attr_penalty_reg*self.attr_penalty)
        self.Loss_Overall    = self.Loss_Latent_L2Dist + self.Loss_Img_L2Dist + self.Loss_Attack + self.Loss_attr + self.Loss_attr_penalty
        # self.Loss_Overall    = self.Loss_Attack

        self.learning_rate = tf.train.polynomial_decay(self.INIT_LEARNING_RATE, self.global_step, self.MAX_ITERATIONS, 0, power=0.5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        start_vars = set(x.name for x in tf.global_variables())
        # colocate_gradients_with_ops places each gradient op on the same
        # device as its forward op. Without it, TF1 puts every gradient on
        # the optimizer's device (GPU here) — which on TF1.15 + A100 hits
        # the cuBLAS-10 SGEMM bug for the celebA ResNet50 backward pass
        # even though predictsym was pinned to CPU.
        self.train = optimizer.minimize(self.Loss_Overall, var_list=[self.adv_latent], global_step=self.global_step, colocate_gradients_with_ops=True)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.orig_img.assign(self.assign_orig_img))
        self.setup.append(self.orig_latent.assign(self.assign_orig_latent))
        self.setup.append(self.adv_latent.assign(self.assign_adv_latent))
        self.setup.append(self.target_lab.assign(self.assign_target_lab))
        self.setup.append(self.const.assign(self.assign_const))
        

        self.init = tf.variables_initializer(var_list=[self.global_step]+[self.adv_latent]+new_vars)

    def attack(self, imgs, labs, latent):
        """
        Find PN for an input instance input_image e.g. celebA is shape (1, 224, 224, 3)
        
        Input:
            imgs (numpy.ndarry): images to be explained, of shape (num_images, size, size, channels)
            labs: one hot encoded vectors of target labels for PN, i.e. which labels
                to explain with an image of a different class
            latent (numpy.ndarry): image to be explained, of shape (1, size, size, channels)
                in the latent space
                
        Output: 
            adv_img (numpy.ndarry): the pertinent negative image
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                # x[y] -= self.kappa if self.PP else -self.kappa
                if self.mode == "PP":
                    x[y] -= self.kappa
                elif self.mode == "PN":
                    x[y] += self.kappa
                x = np.argmax(x)
            if self.mode == "PP":
                return x==y
            else: 
                return x!=y

        batch_size = self.batch_size

        # Populate GAN variables from the safetensors weights once per
        # explainer instance, after the caller's global_variables_initializer
        # has run. Subsequent attack() calls are no-ops.
        if not self._gan_weights_loaded:
            self.sess.run(self.gan_assigns)
            self._gan_weights_loaded = True

        # set the lower and upper bounds accordingly
        Const_LB = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.init_const
        Const_UB = np.ones(batch_size)*1e10
        # the best l2, score, and image attack
        overall_best_dist = [1e10]*batch_size
        overall_best_attack = [np.zeros(imgs[0].shape)]*batch_size

        for binary_search_steps_idx in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            img_batch = imgs[:batch_size]
            label_batch = labs[:batch_size]

            current_step_best_dist = [1e10]*batch_size
            current_step_best_score = [-1]*batch_size

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_orig_img: img_batch,
                                       self.assign_target_lab: label_batch,
                                       self.assign_const: CONST,
                                       self.assign_adv_latent: latent,
                                       self.assign_orig_latent: latent
                                       })

            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack

                self.sess.run([self.train])
                temp_adv_latent = self.sess.run(self.adv_latent)
                self.sess.run(self.adv_updater, feed_dict={self.assign_adv_latent: temp_adv_latent})

                Loss_Overall, OutputScore, adv_img = self.sess.run([self.Loss_Overall, self.ImgToEnforceLabel_Score, self.adv_img])
                Loss_Attack, Loss_Latent_L2Dist, Loss_Img_L2Dist, Loss_attr = self.sess.run([self.Loss_Attack, self.Loss_Latent_L2Dist, self.Loss_Img_L2Dist, self.attr_score])
                target_lab_score, max_nontarget_lab_score_s = self.sess.run([self.target_lab_score, self.max_nontarget_lab_score])


                

                # if iteration%(self.MAX_ITERATIONS//10) == 0:
                if iteration % 10 == 0:
                    print("iter:{} const:{}". format(iteration, CONST))
                    print("Loss_Overall:{:.4f}, Loss_Attack:{:.4f}, Loss_attr:{:.4f}". format(Loss_Overall, Loss_Attack, Loss_attr))
                    print("Loss_Latent_L2Dist:{:.4f}, Loss_Img_L2Dist:{:.4f}". format(Loss_Latent_L2Dist, Loss_Img_L2Dist))
                    print("target_lab_score:{:.4f}, max_nontarget_lab_score:{:.4f}". format(target_lab_score[0], max_nontarget_lab_score_s[0]))
                    print("")
                    sys.stdout.flush()

                for batch_idx,(the_dist, the_score, the_adv_img) in enumerate(zip([Loss_Overall], OutputScore, adv_img)):
                    if the_dist < current_step_best_dist[batch_idx] and compare(the_score, np.argmax(label_batch[batch_idx])):
                        current_step_best_dist[batch_idx] = the_dist
                        current_step_best_score[batch_idx] = np.argmax(the_score)
                    if the_dist < overall_best_dist[batch_idx] and compare(the_score, np.argmax(label_batch[batch_idx])):
                        overall_best_dist[batch_idx] = the_dist
                        overall_best_attack[batch_idx] = the_adv_img

            # adjust the constant as needed
            for batch_idx in range(batch_size):
                if compare(current_step_best_score[batch_idx], np.argmax(label_batch[batch_idx])) and current_step_best_score[batch_idx] != -1:
                    # success, divide const by two
                    Const_UB[batch_idx] = min(Const_UB[batch_idx],CONST[batch_idx])
                    if Const_UB[batch_idx] < 1e9:
                        CONST[batch_idx] = (Const_LB[batch_idx] + Const_UB[batch_idx])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    Const_LB[batch_idx] = max(Const_LB[batch_idx],CONST[batch_idx])
                    if Const_UB[batch_idx] < 1e9:
                        CONST[batch_idx] = (Const_LB[batch_idx] + Const_UB[batch_idx])/2
                    else:
                        CONST[batch_idx] *= 10

        # return the best solution found
        overall_best_attack = overall_best_attack[0]
        return overall_best_attack.reshape((1,) + overall_best_attack.shape)
