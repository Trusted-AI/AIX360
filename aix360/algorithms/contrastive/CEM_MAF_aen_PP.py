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


import sys, os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras.api.keras.models import Model, Sequential, model_from_json
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint

class AEADEN:
    def __init__(self, sess, model, mask_mat, mode, batch_size, kappa, init_learning_rate,
                 binary_search_steps, max_iterations, initial_const, beta, gamma, attributes, aix360_path):
        """
        Initialize PP explainer object. 
        
        Args:
            sess (tensorflow.python.client.session.Session): Tensorflow session
            model: KerasClassifier that contains a trained model to be explained
            mask_mat (numpy.ndarry): Array containing PP masks for each class
            mode (str): "PN" for pertinent negative or "PP" for pertinent positive
            batch_size (int): batch size for how many instances to explain
            kappa (float): Confidence parameter that controls difference between prediction of
                PN (or PP) and original prediction
            init_learning_rate (float): initial learning rate for gradient descent optimizer
            binary_search_steps (int): Controls number of random restarts to find best PN
            max_iterations (int): Max number iterations to run some version of gradient descent on
                PP optimization problem from a single random initialization, i.e., total 
                number of iterations wll be arg_binary_search_steps * arg_max_iterations
            initial_const (int): Constant used for upper/lower bounds in binary search
            gamma (float): Penalty parameter encouraging addition of attributes for PP
            attributes (str list): list of attributes to load attribute classifiers for
            aix360_path (str): path to aix360 used to determine paths to pretrained attribute classifiers 
        """

#        image_size, num_channels, nun_classes = model.image_size, model.num_channels, model.num_labels
        # %%change%%
        image_size = model._input_shape[0]
        num_channels = model._input_shape[2]
        nun_classes = model._nb_classes 
        shape = (batch_size, image_size, image_size, num_channels)
        mask_shape = (batch_size, image_size, image_size, 1)
        mask_num = mask_mat.shape[0]
        mask_vec_shape = (mask_num, 1, 1)
        mask_mat_shape = (mask_num, image_size, image_size)
        self.mask_num = mask_num
        self.sess = sess
        self.INIT_LEARNING_RATE = init_learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.kappa = kappa
        self.init_const = initial_const
        self.batch_size = batch_size
        self.AE = None
        self.mode = mode
        self.beta = beta
        self.gamma = gamma
        self.attributes = attributes
        self.aix360_path = aix360_path
        
        ### Load attribute classifier
        nn_type = "simple"
        #import copy
        attr_model_list=[]
        for attr in self.attributes:
            # load test data into memory using Image Data Generator
#            print("Loading data for " + attr + " into memory")
            # load json and create model
            json_file_name = os.path.join(aix360_path, "models/CEM_MAF/{}_{}_model.json".format(nn_type, attr))
            json_file = open(json_file_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            weight_file_name = os.path.join(aix360_path, "models/CEM_MAF/{}_{}_weights.h5".format(nn_type, attr))
            loaded_model.load_weights(weight_file_name)
            print("Loaded model for " + attr + " from disk")
            attr_model_list.append(loaded_model)
        
        print("# of attr models is",len(attr_model_list))


#        print("beta:{}".format(self.beta))
        # these are variables to be more efficient in sending data to tf
        self.orig_img = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.mask_vec = tf.Variable(np.zeros(mask_vec_shape), dtype=tf.float32)
        self.mask_vec_s = tf.Variable(np.zeros(mask_vec_shape), dtype=tf.float32)
        self.mask_mat = tf.constant(mask_mat, dtype=tf.float32)
        # self.img_mask = tf.Variable(np.zeros(mask_shape), dtype=tf.float32)
        # self.img_mask_s = tf.Variable(np.zeros(mask_shape), dtype=tf.float32, name="var_mask_s")
        self.target_lab = tf.Variable(np.zeros((batch_size,nun_classes)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.global_step = tf.Variable(0.0, trainable=False)

        # and here's what we use to assign them
        self.assign_orig_img = tf.placeholder(tf.float32, shape)
        self.assign_mask_vec = tf.placeholder(tf.float32, mask_vec_shape)
        self.assign_mask_vec_s = tf.placeholder(tf.float32, mask_vec_shape)
        # self.assign_img_mask = tf.placeholder(tf.float32, mask_shape)
        # self.assign_img_mask_s = tf.placeholder(tf.float32, mask_shape)
        self.assign_target_lab = tf.placeholder(tf.float32, (batch_size,nun_classes))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])


        """Fast Iterative Soft Thresholding"""
        """--------------------------------"""
        

        # self.zt = tf.divide(self.global_step, self.global_step+tf.cast(3, tf.float32))
        # cond1 = tf.cast(tf.greater(tf.subtract(self.adv_img_mask_s, self.orig_img),self.beta), tf.float32)
        # cond2 = tf.cast(tf.less_equal(tf.abs(tf.subtract(self.adv_img_s,self.orig_img)),self.beta), tf.float32)
        # cond3 = tf.cast(tf.less(tf.subtract(self.adv_img_s, self.orig_img),tf.negative(self.beta)), tf.float32)
        # upper = tf.minimum(tf.subtract(self.adv_img_s, self.beta), tf.cast(0.5, tf.float32))
        # lower = tf.maximum(tf.add(self.adv_img_s, self.beta), tf.cast(-0.5, tf.float32))
        # self.assign_adv_img = tf.multiply(cond1,upper)+tf.multiply(cond2,self.orig_img)+tf.multiply(cond3,lower)

        self.zt = tf.divide(self.global_step, self.global_step+tf.cast(3, tf.float32))
        """
        x = x - beta if x > beta
        x = 0 if x < beta
        """
        cond1 = tf.cast(tf.greater(self.mask_vec_s, self.beta), tf.float32)
        cond2 = tf.cast(tf.less_equal(self.mask_vec_s, self.beta), tf.float32)
        upper = tf.minimum(tf.subtract(self.mask_vec_s, self.beta), tf.cast(1, tf.float32))
        self.assign_mask_vec = tf.multiply(cond1,upper) + tf.multiply(cond2, tf.constant(0, tf.float32))
        self.assign_mask_vec_s = self.assign_mask_vec+tf.multiply(self.zt, self.assign_mask_vec-self.mask_vec)
        # cond4=tf.cast(tf.greater(tf.subtract( self.assign_adv_img, self.orig_img),0), tf.float32)
        # cond5=tf.cast(tf.less_equal(tf.subtract( self.assign_adv_img,self.orig_img),0), tf.float32)
        # if self.mode == "PP":
        #     self.assign_adv_img = tf.multiply(cond5,self.assign_adv_img)+tf.multiply(cond4,self.orig_img)
        # elif self.mode == "PN":
        #     self.assign_adv_img = tf.multiply(cond4,self.assign_adv_img)+tf.multiply(cond5,self.orig_img)

        # self.assign_adv_img_s = self.assign_adv_img+tf.multiply(self.zt, self.assign_adv_img-self.adv_img)
        # cond6=tf.cast(tf.greater(tf.subtract(  self.assign_adv_img_s, self.orig_img),0), tf.float32)
        # cond7=tf.cast(tf.less_equal(tf.subtract(  self.assign_adv_img_s,self.orig_img),0), tf.float32)
        # if self.mode == "PP":
        #     self.assign_adv_img_s = tf.multiply(cond7, self.assign_adv_img_s)+tf.multiply(cond6,self.orig_img)
        # elif self.mode == "PN":
        #     self.assign_adv_img_s = tf.multiply(cond6, self.assign_adv_img_s)+tf.multiply(cond7,self.orig_img)
        self.mask_updater = tf.assign(self.mask_vec, self.assign_mask_vec)
        self.mask_updater_s = tf.assign(self.mask_vec_s, self.assign_mask_vec_s)
        """ Thresholding """

        # mask_ones = tf.constant(1, tf.float32)
        # mask_zeros = tf.constant(0, tf.float32)

        # mask_cond1 = tf.cast(tf.greater(self.img_mask, 0.5), tf.float32)
        # mask_cond2 = tf.cast(tf.less_equal(self.img_mask, 0.5), tf.float32)
        # self.img_mask_threshold = tf.multiply(mask_cond1, mask_ones)+tf.multiply(mask_cond2, mask_zeros)
        # cannot find gradient
        self.img_mask = tf.reduce_sum(self.mask_vec * self.mask_mat, axis=0)
        self.img_mask = tf.expand_dims(self.img_mask, axis=2)
        self.img_mask = tf.expand_dims(self.img_mask, axis=0)
        self.adv_img = tf.multiply(self.img_mask, self.orig_img)


        # mask_s_cond1 = tf.cast(tf.greater(self.img_mask_s, 0.5), tf.float32)
        # mask_s_cond2 = tf.cast(tf.less_equal(self.img_mask_s, 0.5), tf.float32)
        # self.img_mask_threshold_s = tf.multiply(mask_s_cond1, mask_ones)+tf.multiply(mask_s_cond2, mask_zeros)

        # cannot find gradient
        self.img_mask_s = tf.reduce_sum(self.mask_vec_s * self.mask_mat, axis=0)
        self.img_mask_s = tf.expand_dims(self.img_mask_s, axis=2)
        self.img_mask_s = tf.expand_dims(self.img_mask_s, axis=0)
        self.adv_img_s = tf.multiply(self.img_mask_s, self.orig_img)

        
        """--------------------------------"""
        # prediction from attribute classifer
        self.delta_img = self.orig_img-self.adv_img
        self.delta_img_s = self.orig_img-self.adv_img_s
        if self.mode == "PP":
            self.attr_score = tf.constant(0, dtype="float32")
            self.attr_score_s = tf.constant(0, dtype="float32")
            #print(attr_model_list[0].predict(self.adv_img)) 
            #print(loaded_model.predict(self.adv_img)) 
            for i in range(len(attr_model_list)):
                self.attr_score = self.attr_score + tf.maximum(attr_model_list[i](self.adv_img) - attr_model_list[i](self.orig_img),tf.constant(0, tf.float32))
                self.attr_score_s = self.attr_score_s + tf.maximum(attr_model_list[i](self.adv_img_s) - attr_model_list[i](self.orig_img),tf.constant(0, tf.float32))
                #print(self.attr_score.shape) 
            self.attr_score = tf.squeeze(self.attr_score)
            self.attr_score_s = tf.squeeze(self.attr_score_s)  
           #self.ImgToEnforceLabel_Score = model.predict(self.adv_img)
            #self.ImgToEnforceLabel_Score_s = model.predict(self.adv_img_s)
# %%change%%
        elif self.mode == "PN":
#            self.ImgToEnforceLabel_Score = model.predict(self.adv_img)
#            self.ImgToEnforceLabel_Score_s = model.predict(self.adv_img_s)
            self.ImgToEnforceLabel_Score = model.predictsym(self.adv_img)
            self.ImgToEnforceLabel_Score_s = model.predictsym(self.adv_img_s)

        # prediction BEFORE-SOFTMAX of the model
        self.delta_img = self.orig_img-self.adv_img
        self.delta_img_s = self.orig_img-self.adv_img_s
        if self.mode == "PP":
#            self.ImgToEnforceLabel_Score = model.predict(self.adv_img)
#            self.ImgToEnforceLabel_Score_s = model.predict(self.adv_img_s)
            self.ImgToEnforceLabel_Score = model.predictsym(self.adv_img)
            self.ImgToEnforceLabel_Score_s = model.predictsym(self.adv_img_s)
        elif self.mode == "PN":
#            self.ImgToEnforceLabel_Score = model.predict(self.adv_img)
#            self.ImgToEnforceLabel_Score_s = model.predict(self.adv_img_s)
            self.ImgToEnforceLabel_Score = model.predictsym(self.adv_img)
            self.ImgToEnforceLabel_Score_s = model.predictsym(self.adv_img_s)

        # distance to the input data
        self.L2_dist = tf.reduce_sum(tf.square(self.img_mask),[1,2,3])
        self.L2_dist_s = tf.reduce_sum(tf.square(self.img_mask_s),[1,2,3])
        self.L1_dist = tf.reduce_sum(tf.abs(self.img_mask),[1,2,3])
        self.L1_dist_s = tf.reduce_sum(tf.abs(self.img_mask_s),[1,2,3])
        self.EN_dist = self.L2_dist + tf.multiply(self.L1_dist, self.beta)
        self.EN_dist_s = self.L2_dist_s + tf.multiply(self.L1_dist_s, self.beta)

        # compute the probability of the label class versus the maximum other
        self.target_lab_score        = tf.reduce_sum((self.target_lab)*self.ImgToEnforceLabel_Score,1)
        target_lab_score_s           = tf.reduce_sum((self.target_lab)*self.ImgToEnforceLabel_Score_s,1)
        self.max_nontarget_lab_score = tf.reduce_max((1-self.target_lab)*self.ImgToEnforceLabel_Score - (self.target_lab*10000),1)
        max_nontarget_lab_score_s    = tf.reduce_max((1-self.target_lab)*self.ImgToEnforceLabel_Score_s - (self.target_lab*10000),1)
        if self.mode == "PP":
            Loss_Attack = tf.maximum(0.0, self.max_nontarget_lab_score - self.target_lab_score + self.kappa)
            Loss_Attack_s = tf.maximum(0.0, max_nontarget_lab_score_s - target_lab_score_s + self.kappa)
        elif self.mode == "PN":
            Loss_Attack = tf.maximum(0.0, -self.max_nontarget_lab_score + self.target_lab_score + self.kappa)
            Loss_Attack_s = tf.maximum(0.0, -max_nontarget_lab_score_s + target_lab_score_s + self.kappa)
        # sum up the losses
        self.Loss_L1Dist    = tf.reduce_sum(self.L1_dist)
        self.Loss_L1Dist_s  = tf.reduce_sum(self.L1_dist_s)
        self.Loss_L2Dist    = tf.reduce_sum(self.L2_dist)
        self.Loss_L2Dist_s  = tf.reduce_sum(self.L2_dist_s)
        self.Loss_Attack    = tf.reduce_sum(self.const*Loss_Attack)
        with tf.name_scope("loss_attack_s"):
            self.Loss_Attack_s  = tf.reduce_sum(self.const*Loss_Attack_s)
        if self.AE:
            if self.mode == "PP":
                self.Loss_AE_Dist   = self.gamma*tf.square(tf.norm(self.AE(self.delta_img)-self.delta_img))
                self.Loss_AE_Dist_s = self.gamma*tf.square(tf.norm(self.AE(self.delta_img)-self.delta_img_s))
            elif self.mode == "PN":
                self.Loss_AE_Dist   = self.gamma*tf.square(tf.norm(self.AE(self.adv_img)-self.adv_img))
                self.Loss_AE_Dist_s = self.gamma*tf.square(tf.norm(self.AE(self.adv_img_s)-self.adv_img_s))
        else:
            self.Loss_AE_Dist = tf.constant(0, dtype="float32")
            self.Loss_AE_Dist_s = tf.constant(0, dtype="float32")

        with tf.name_scope("loss"):
            # self.Loss_ToOptimize = self.Loss_Attack_s + self.Loss_L2Dist_s + self.Loss_AE_Dist_s
            self.Loss_ToOptimize = self.Loss_Attack_s + tf.multiply(self.gamma,self.attr_score_s)
        #self.Loss_Overall    = self.Loss_Attack   + self.Loss_L2Dist   + self.Loss_AE_Dist   + tf.multiply(self.beta, self.Loss_L1Dist)
        self.Loss_Overall    = self.Loss_Attack  + tf.multiply(self.gamma,self.attr_score_s)   + tf.multiply(self.beta, self.Loss_L1Dist)
        #print(self.Loss_Attack.shape, self.Loss_Overall.shape)
        self.learning_rate = tf.train.polynomial_decay(self.INIT_LEARNING_RATE, self.global_step, self.MAX_ITERATIONS, 0, power=0.5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        start_vars = set(x.name for x in tf.global_variables())
        # self.train = optimizer.minimize(self.Loss_ToOptimize, var_list=[self.adv_img_s], global_step=self.global_step)
        self.train = optimizer.minimize(self.Loss_ToOptimize, var_list=[self.mask_vec_s], global_step=self.global_step)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.orig_img.assign(self.assign_orig_img))
        self.setup.append(self.target_lab.assign(self.assign_target_lab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.mask_vec.assign(self.assign_mask_vec))
        self.setup.append(self.mask_vec_s.assign(self.assign_mask_vec_s))

        self.init = tf.variables_initializer(var_list=[self.global_step]+[self.mask_vec_s]+[self.mask_vec]+new_vars)

    def attack(self, imgs, labs):
        """
        Find PN for an input instance input_image e.g. celebA is shape (1, 224, 224, 3)
        
        Input:
            imgs (numpy.ndarry): images to be explained, of shape (num_images, size, size, channels)
            labs: one hot encoded vectors of target label for original image prediction
            latent (numpy.ndarry): image to be explained, of shape (1, size, size, channels)
                in the latent space
                
        Output: 
            adv_img (numpy.ndarry): the pertinent positive image
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

        # set the lower and upper bounds accordingly
        Const_LB = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.init_const
        Const_UB = np.ones(batch_size)*1e10
        # the best l2, score, and image attack
        overall_best_dist = [1e10]*batch_size
        overall_best_attack = [np.zeros(imgs[0].shape)]*batch_size
        overall_best_mask_vec = [np.zeros((self.mask_num, 1, 1))]*batch_size

        for binary_search_steps_idx in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            img_batch = imgs[:batch_size]
            label_batch = labs[:batch_size]
            img_shape = img_batch.shape
            img_mask_vec = np.ones((self.mask_num, 1, 1))

            current_step_best_dist = [1e10]*batch_size
            current_step_best_score = [-1]*batch_size

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_orig_img: img_batch,
                                       self.assign_target_lab: label_batch,
                                       self.assign_const: CONST,
                                       self.assign_mask_vec: img_mask_vec,
                                       self.assign_mask_vec_s: img_mask_vec})



            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                self.sess.run([self.train])
                self.sess.run([self.mask_updater, self.mask_updater_s])

                Loss_Overall, Loss_EN, OutputScore, adv_img, img_mask = self.sess.run([self.Loss_Overall, self.EN_dist, self.ImgToEnforceLabel_Score, self.adv_img, self.mask_vec])
                # print("max:{}, min:{}".format(np.max(img_mask), np.min(img_mask)))
                Loss_Attack, Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist, Loss_attr = self.sess.run([self.Loss_Attack, self.Loss_L2Dist, self.Loss_L1Dist, self.Loss_AE_Dist, self.attr_score])
                target_lab_score, max_nontarget_lab_score_s = self.sess.run([self.target_lab_score, self.max_nontarget_lab_score])
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print("iter:{} const:{}". format(iteration, CONST))
                    print("Loss_Overall:{:.4f}, Loss_Attack:{:.4f}, Loss_attr:{:.4f}". format(Loss_Overall, Loss_Attack, Loss_attr))
                    print("Loss_L2Dist:{:.4f}, Loss_L1Dist:{:.4f}, AE_loss:{}". format(Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist))
                    print("target_lab_score:{:.4f}, max_nontarget_lab_score:{:.4f}". format(target_lab_score[0], max_nontarget_lab_score_s[0]))
                    print("")
                    sys.stdout.flush()

                for batch_idx,(the_dist, the_score, the_adv_img, the_mask) in enumerate(zip(Loss_EN, OutputScore, adv_img, img_mask)):
                    if the_dist < current_step_best_dist[batch_idx] and compare(the_score, np.argmax(label_batch[batch_idx])):
                        current_step_best_dist[batch_idx] = the_dist
                        current_step_best_score[batch_idx] = np.argmax(the_score)
                    if the_dist < overall_best_dist[batch_idx] and compare(the_score, np.argmax(label_batch[batch_idx])):
                        overall_best_dist[batch_idx] = the_dist
                        overall_best_attack[batch_idx] = the_adv_img
                        overall_best_mask_vec[batch_idx] = img_mask

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
        overall_best_mask_vec = overall_best_mask_vec[0]
        # overall_best_mask_vec = overall_best_mask_vec.reshape(-1)
        return overall_best_attack.reshape((1,) + overall_best_attack.shape), overall_best_mask_vec

        """
        export PYTHONPATH=$PYTHONPATH:/u/pinyu/progressive_growing_of_gans
        python3 gen_example.py -s 6

        """
    
    def generate_PP(self, img_mask, orig_img, orig_class, model, mask_size, mask_mat):
        def model_prediction(model, inputs):
            prob = model.model.predict(inputs)
            predicted_class = np.argmax(prob)
            prob_str = np.array2string(prob).replace('\n','')
            return prob, predicted_class, prob_str        # ranking

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
            adv_img = working_mask * orig_img
            img_prob, img_class, img_prob_str = model_prediction(model, adv_img)
            print("i:{}, index:{}, value:{}, class:{}".format(i, temp_index, mask_vec[temp_index], img_class))
            if img_class == orig_class:
                success = True
                break
                
        return adv_img, success
