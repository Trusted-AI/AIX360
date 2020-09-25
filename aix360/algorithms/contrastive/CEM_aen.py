import sys
import numpy as np
import tensorflow as tf
import keras.backend as K


class AEADEN:

    def __init__(self, model, shape, mode, AE, batch_size, kappa, init_learning_rate,
                 binary_search_steps, max_iterations, initial_const, beta, gamma, alpha=0, threshold=1, offset=0):
        
        """
        Constructor method.
        
        Args:
            model: KerasClassifier classification model
            arg_mode(str): 'PP' or 'PN'
            AE: Auto-encoder model
            batch_size(int): Number of samples in a batch
            kappa(double): Confidence gap between desired class and other classes
            init_learning_rate(double): Initial learning rate
            binary_search_steps(int): Number of search steps
            max_iterations(int): For each weighting of loss function number of iterations to search
            initial_const(double): Initial weighting of loss function
            beta (double): Weighting of L1 loss
            gamma (double): Weighting of auto-encoder
            alpha (double): Weighting of L2 loss
            threshold (double): automatically turn off all features less than threshold since nothing to turn off
            offset (double): example is in [0,1]. we subtract offset when passed to classifier
        """

        num_classes = model._nb_classes
        tf_sum = list(range(1, len(shape)))

        self.sess = K.get_session()

        self.INIT_LEARNING_RATE = init_learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.kappa = kappa
        self.init_const = initial_const
        self.batch_size = batch_size
        self.AE = AE
        self.mode = mode
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha 
        self.offset = offset
        self.threshold = threshold

        # these are variables to be more efficient in sending data to tf
        self.orig_img = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.adv_img = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.adv_img_s = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.target_lab = tf.Variable(np.zeros((batch_size, num_classes)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.global_step = tf.Variable(0.0, trainable=False)

        # and here's what we use to assign them
        self.assign_orig_img = tf.placeholder(tf.float32, shape)
        self.assign_adv_img = tf.placeholder(tf.float32, shape)
        self.assign_adv_img_s = tf.placeholder(tf.float32, shape)
        self.assign_target_lab = tf.placeholder(tf.float32, (batch_size, num_classes))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])

        """Fast Iterative Soft Thresholding"""
        """--------------------------------"""
        self.zt = tf.divide(self.global_step, self.global_step + tf.cast(3, tf.float32))

        # These conditions do shrinkage threshold - note that sparsity is on orig_img-adv_img_s so a change of variables just shrinkage on adv_img - orig_img
        cond1 = tf.cast(tf.greater(tf.subtract(self.adv_img_s, self.orig_img), self.beta), tf.float32)
        cond2 = tf.cast(tf.less_equal(tf.abs(tf.subtract(self.adv_img_s, self.orig_img)), self.beta), tf.float32)
        cond3 = tf.cast(tf.less(tf.subtract(self.adv_img_s, self.orig_img), tf.negative(self.beta)), tf.float32)
#        upper = tf.minimum(tf.subtract(self.adv_img_s, self.beta), tf.cast(0.5, tf.float32))
#        lower = tf.maximum(tf.add(self.adv_img_s, self.beta), tf.cast(-0.5, tf.float32))
        upper = tf.subtract(self.adv_img_s, self.beta) # RL: removed the bounds
        lower = tf.add(self.adv_img_s, self.beta)
        self.assign_adv_img = tf.multiply(cond1, upper) + tf.multiply(cond2, self.orig_img) + tf.multiply(cond3, lower)
            
        if self.mode == "PP":
            cond8 = tf.cast(tf.greater(tf.subtract(self.orig_img, self.assign_adv_img), tf.cast(1.0, tf.float32)), tf.float32)
            cond9 = tf.cast(tf.less(tf.subtract(self.orig_img, self.assign_adv_img), tf.cast(0.0, tf.float32)), tf.float32)
            cond10 = tf.cast(tf.logical_and(tf.greater_equal(tf.subtract(self.orig_img, self.assign_adv_img), tf.cast(0.0, tf.float32)), tf.less_equal(tf.subtract(self.orig_img, self.assign_adv_img), tf.cast(1.0, tf.float32))), tf.float32)
            self.assign_adv_img = tf.multiply(cond8, tf.subtract(self.orig_img, tf.cast(1.0, tf.float32))) + tf.multiply(cond9, self.orig_img) + tf.multiply(cond10, self.assign_adv_img)
        elif self.mode == "PN":
            cond8_pn = tf.cast(tf.greater(self.assign_adv_img, tf.cast(1.0, tf.float32)), tf.float32)
            cond9_pn = tf.cast(tf.less(self.assign_adv_img, tf.cast(0.0, tf.float32)), tf.float32)
            cond10_pn = tf.cast(tf.logical_and(tf.greater_equal(self.assign_adv_img, tf.cast(0.0, tf.float32)), tf.less_equal(self.assign_adv_img, tf.cast(1.0, tf.float32))), tf.float32)
            self.assign_adv_img = tf.multiply(cond8_pn, tf.cast(1.0, tf.float32)) + tf.multiply(cond9_pn, tf.cast(0.0, tf.float32)) + tf.multiply(cond10_pn, self.assign_adv_img)
        
        # if self.threshold < 1, use it to turn off features less than the threshold
        if (self.mode == "PP") and (self.threshold < 1.0):
            cond_thresh1a = tf.cast(tf.less(tf.subtract(self.orig_img, self.assign_adv_img), tf.cast(self.threshold, tf.float32)), tf.float32)
            cond_thresh1b = tf.cast(tf.greater_equal(tf.subtract(self.orig_img, self.assign_adv_img), tf.cast(self.threshold, tf.float32)), tf.float32)
            self.assign_adv_img = tf.multiply(cond_thresh1b, self.assign_adv_img) + tf.multiply(cond_thresh1a, self.orig_img) # theshold delta to 0 if <= self.threshold
        elif self.mode == "PN" and (self.threshold < 1.0):
            cond_thresh1a = tf.cast(tf.greater_equal(self.assign_adv_img, tf.cast(self.threshold, tf.float32)), tf.float32)
            self.assign_adv_img = tf.multiply(cond_thresh1a, self.assign_adv_img)
            
        if self.mode == "PP":    
            self.assign_adv_img = tf.maximum(self.assign_adv_img, tf.cast(0.0, tf.float32)) # PP's cannot increase a data point
        elif self.mode == "PN":
            self.assign_adv_img = tf.maximum(self.assign_adv_img, self.orig_img) # PN's cannot remove part of a data point

        # Removing cond4 and cond5 since no more projection needed here
        # These conditions threshold the variable after shrinkage to the proper space
        # cond4 = tf.cast(tf.greater(tf.subtract(self.assign_adv_img, self.orig_img), 0), tf.float32)
        # cond5 = tf.cast(tf.less_equal(tf.subtract(self.assign_adv_img, self.orig_img), 0), tf.float32)
        # if self.mode == "PP" or self.mode == "PP_PATH":
        #     self.assign_adv_img = tf.multiply(cond5, self.assign_adv_img) + tf.multiply(cond4, self.orig_img)
        # elif self.mode == "PN":
        #     self.assign_adv_img = tf.multiply(cond4, self.assign_adv_img) + tf.multiply(cond5, self.orig_img)

        # This is how to take a step in FISTA
        self.assign_adv_img_s = self.assign_adv_img + tf.multiply(self.zt, self.assign_adv_img - self.adv_img)

        # Replacing cond6 and cond7 with same projections as before FISTA step but for self.asign_adv_img_s
        if self.mode == "PP":    
            cond11 = tf.cast(tf.greater(tf.subtract(self.orig_img, self.assign_adv_img_s), tf.cast(1.0, tf.float32)), tf.float32)
            cond12 = tf.cast(tf.less(tf.subtract(self.orig_img, self.assign_adv_img_s), tf.cast(0.0, tf.float32)), tf.float32)
            cond13 = tf.cast(tf.logical_and(tf.greater_equal(tf.subtract(self.orig_img, self.assign_adv_img_s), tf.cast(0.0, tf.float32)), tf.less_equal(tf.subtract(self.orig_img, self.assign_adv_img_s), tf.cast(1.0, tf.float32))), tf.float32)
            self.assign_adv_img_s = tf.multiply(cond11, tf.subtract(self.orig_img, tf.cast(1.0, tf.float32))) + tf.multiply(cond12, self.orig_img) + tf.multiply(cond13, self.assign_adv_img_s)
        elif self.mode == "PN":
            cond11_pn = tf.cast(tf.greater(self.assign_adv_img_s, tf.cast(1.0, tf.float32)), tf.float32)
            cond12_pn = tf.cast(tf.less(self.assign_adv_img_s, tf.cast(0.0, tf.float32)), tf.float32)
            cond13_pn = tf.cast(tf.logical_and(tf.greater_equal(self.assign_adv_img_s, tf.cast(0.0, tf.float32)), tf.less_equal(self.assign_adv_img_s, tf.cast(1.0, tf.float32))), tf.float32)
            self.assign_adv_img_s = tf.multiply(cond11_pn, tf.cast(1.0, tf.float32)) + tf.multiply(cond12_pn, tf.cast(0.0, tf.float32)) + tf.multiply(cond13_pn, self.assign_adv_img_s)
            
        # if self.threshold < 1, use it to turn off features less than the threshold
        if (self.mode == "PP") and (self.threshold < 1.0):
            cond_thresh2a = tf.cast(tf.less(tf.subtract(self.orig_img, self.assign_adv_img_s), tf.cast(self.threshold, tf.float32)), tf.float32)
            cond_thresh2b = tf.cast(tf.greater_equal(tf.subtract(self.orig_img, self.assign_adv_img_s), tf.cast(self.threshold, tf.float32)), tf.float32)
            self.assign_adv_img_s = tf.multiply(cond_thresh2b, self.assign_adv_img) + tf.multiply(cond_thresh2a, self.orig_img) # theshold delta to 0 if <= self.threshold
        elif self.mode == "PN" and (self.threshold < 1.0):
            cond_thresh1a = tf.cast(tf.greater_equal(self.assign_adv_img_s, tf.cast(self.threshold, tf.float32)), tf.float32)
            self.assign_adv_img_s = tf.multiply(cond_thresh1a, self.assign_adv_img_s)

        if self.mode == "PP":    
            self.assign_adv_img_s = tf.maximum(self.assign_adv_img_s, tf.cast(0.0, tf.float32)) # PP's cannot increase a data point
        elif self.mode == "PN":
            self.assign_adv_img_s = tf.maximum(self.assign_adv_img_s, self.orig_img) # PN's cannot remove part of a data point

        # These conditions threshold the variable after taking a step in FISTA
        # cond6 = tf.cast(tf.greater(tf.subtract(self.assign_adv_img_s, self.orig_img), 0), tf.float32)
        # cond7 = tf.cast(tf.less_equal(tf.subtract(self.assign_adv_img_s, self.orig_img), 0), tf.float32)
        # if self.mode == "PP" or self.mode == "PP_PATH":
        #     self.assign_adv_img_s = tf.multiply(cond7, self.assign_adv_img_s) + tf.multiply(cond6, self.orig_img)
        # elif self.mode == "PN":
        #     self.assign_adv_img_s = tf.multiply(cond6, self.assign_adv_img_s) + tf.multiply(cond7, self.orig_img)

        self.adv_updater = tf.assign(self.adv_img, self.assign_adv_img)
        self.adv_updater_s = tf.assign(self.adv_img_s, self.assign_adv_img_s)

        """--------------------------------"""
        # prediction BEFORE-SOFTMAX of the model
        self.delta_img = self.orig_img - self.adv_img
        self.delta_img_s = self.orig_img - self.adv_img_s
        # %%change%%
        if self.mode == "PP":
            # self.ImgToEnforceLabel_Score = model.predict(self.delta_img)
            # self.ImgToEnforceLabel_Score_s = model.predict(self.delta_img_s)
            self.ImgToEnforceLabel_Score = model.predictsym(tf.subtract(self.delta_img, tf.cast(self.offset, tf.float32)))
            self.ImgToEnforceLabel_Score_s = model.predictsym(tf.subtract(self.delta_img_s, tf.cast(self.offset, tf.float32)))
        elif self.mode == "PN":
            # self.ImgToEnforceLabel_Score = model.predict(self.adv_img)
            # self.ImgToEnforceLabel_Score_s = model.predict(self.adv_img_s)
            self.ImgToEnforceLabel_Score = model.predictsym(tf.subtract(self.adv_img, tf.cast(self.offset, tf.float32)))
            self.ImgToEnforceLabel_Score_s = model.predictsym(tf.subtract(self.adv_img_s, tf.cast(self.offset, tf.float32)))

        self.L2_dist = tf.reduce_sum(tf.square(self.delta_img), axis=tf_sum)
        self.L2_dist_s = tf.reduce_sum(tf.square(self.delta_img_s), axis=tf_sum)
        self.L1_dist = tf.reduce_sum(tf.abs(self.delta_img), axis=tf_sum)
        self.L1_dist_s = tf.reduce_sum(tf.abs(self.delta_img_s), axis=tf_sum)
            
        self.EN_dist = tf.multiply(self.L2_dist, self.alpha) + self.L2_dist + tf.multiply(self.L1_dist, self.beta)
        self.EN_dist_s = tf.multiply(self.L2_dist_s, self.alpha) + tf.multiply(self.L1_dist_s, self.beta)
    
        # compute the probability of the label class versus the maximum other
        self.target_lab_score = tf.reduce_sum((self.target_lab) * self.ImgToEnforceLabel_Score, 1)
        target_lab_score_s = tf.reduce_sum((self.target_lab) * self.ImgToEnforceLabel_Score_s, 1)
        self.max_nontarget_lab_score = tf.reduce_max((1 - self.target_lab) * self.ImgToEnforceLabel_Score -
                                                     (self.target_lab * 10000), 1)
        max_nontarget_lab_score_s = tf.reduce_max((1 - self.target_lab) * self.ImgToEnforceLabel_Score_s -
                                                  (self.target_lab * 10000), 1)
        if self.mode == "PP":
            Loss_Attack = tf.maximum(0.0, self.max_nontarget_lab_score - self.target_lab_score + self.kappa)
            Loss_Attack_s = tf.maximum(0.0, max_nontarget_lab_score_s - target_lab_score_s + self.kappa)
        elif self.mode == "PN":
            Loss_Attack = tf.maximum(0.0, -self.max_nontarget_lab_score + self.target_lab_score + self.kappa)
            Loss_Attack_s = tf.maximum(0.0, -max_nontarget_lab_score_s + target_lab_score_s + self.kappa)
        # sum up the losses
        self.Loss_L1Dist = tf.reduce_sum(self.L1_dist)
        self.Loss_L1Dist_s = tf.reduce_sum(self.L1_dist_s)
        self.Loss_L2Dist = tf.reduce_sum(self.L2_dist)
        self.Loss_L2Dist_s = tf.reduce_sum(self.L2_dist_s)
        self.Loss_Attack = tf.reduce_sum(self.const * Loss_Attack)
        self.Loss_Attack_s = tf.reduce_sum(self.const * Loss_Attack_s)        

        if self.mode == "PP" and callable(self.AE):
            self.Loss_AE_Dist = self.gamma * tf.square(tf.norm(self.AE(self.delta_img) - self.delta_img))
            self.Loss_AE_Dist_s = self.gamma * tf.square(tf.norm(self.AE(self.delta_img) - self.delta_img_s))
        elif self.mode == "PN" and callable(self.AE):
            self.Loss_AE_Dist = self.gamma * tf.square(tf.norm(self.AE(self.adv_img) - self.adv_img))
            self.Loss_AE_Dist_s = self.gamma * tf.square(tf.norm(self.AE(self.adv_img_s) - self.adv_img_s))
        else:
            self.Loss_AE_Dist = tf.constant(0.)
            self.Loss_AE_Dist_s = tf.constant(0.)

        self.Loss_ToOptimize = self.Loss_Attack_s + self.Loss_L2Dist_s + self.Loss_AE_Dist_s
        self.Loss_Overall = self.Loss_Attack + self.Loss_L2Dist + self.Loss_AE_Dist + tf.multiply(self.beta,
                                                                                                self.Loss_L1Dist)

        self.learning_rate = tf.train.polynomial_decay(self.INIT_LEARNING_RATE, self.global_step, self.MAX_ITERATIONS,
                                                       0,
                                                       power=0.5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        start_vars = set(x.name for x in tf.global_variables())
        self.train = optimizer.minimize(self.Loss_ToOptimize, var_list=[self.adv_img_s], global_step=self.global_step) # gradient of g(delta) in paper is subtracted from self.adv_img_s
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.orig_img.assign(self.assign_orig_img))
        self.setup.append(self.target_lab.assign(self.assign_target_lab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.adv_img.assign(self.assign_adv_img))
        self.setup.append(self.adv_img_s.assign(self.assign_adv_img_s))

        self.init = tf.variables_initializer(var_list=[self.global_step] + [self.adv_img_s] + [self.adv_img] + new_vars)

    def attack(self, imgs, labs):
        """
        Perturbations to generate explanations
        """
        
        def compare(x,y):
            """
            compare x and y
            """
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x = x - np.max(x) # to prevent overflow
                x = np.exp(x) / np.sum(np.exp(x)) # RL added
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
                                       self.assign_adv_img: img_batch,
                                       self.assign_adv_img_s: img_batch})
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack

                self.sess.run([self.train]) # run a gradient step on g() function (see eq 5 in CEM paper)
                self.sess.run([self.adv_updater, self.adv_updater_s]) # compute shrinkage operation and FISTA step (eqs 5 and 6 in CEM paper)
                                
                Loss_Overall, Loss_EN, OutputScore, adv_img = self.sess.run([self.Loss_Overall, 
                                                                             self.EN_dist, self.ImgToEnforceLabel_Score,
                                                                             self.adv_img])
                                   
                Loss_Attack, Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist = self.sess.run([self.Loss_Attack, self.Loss_L2Dist, 
                                                                                     self.Loss_L1Dist, self.Loss_AE_Dist])                       
                

                target_lab_score, max_nontarget_lab_score_s = self.sess.run([self.target_lab_score, 
                                                                             self.max_nontarget_lab_score])

                                    # %%change%% 
                if iteration%(self.MAX_ITERATIONS//2) == 0:
                    print("iter:{} const:{}". format(iteration, CONST))
                    print("Loss_Overall:{:.4f}, Loss_Attack:{:.4f}". format(Loss_Overall, Loss_Attack))
                    print("Loss_L2Dist:{:.4f}, Loss_L1Dist:{:.4f}, AE_loss:{}". format(Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist))
                    print("target_lab_score:{:.4f}, max_nontarget_lab_score:{:.4f}". format(target_lab_score[0], 
                                                                                            max_nontarget_lab_score_s[0]))
                    print("")
                    sys.stdout.flush()
 
                for batch_idx,(the_dist, the_score, the_adv_img) in enumerate(zip(Loss_EN, OutputScore, adv_img)):
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