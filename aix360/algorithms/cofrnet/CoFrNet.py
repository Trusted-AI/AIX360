'''
author: @ishapuri101
'''

import sys
import pandas as pd #loading data in table form  
import numpy as np # linear algebra
import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from aix360.algorithms.cofrnet.utils import generate_connections
from aix360.algorithms.cofrnet.utils import process_data
from aix360.algorithms.cofrnet.CustomizedLinearClasses import CustomizedLinearFunction
from aix360.algorithms.cofrnet.CustomizedLinearClasses import CustomizedLinear

from aix360.algorithms.die import DIExplainer



        
class CoFrNet_Model(nn.Module):
    """
    CoFrNet_Model is the base class for Continued Fractions Nets (CoFrNets).

    References:
        .. [#1] CoFrNets: Interpretable Neural Architecture Inspired by Continued Fractions, NeurIPS 2021. 
        Isha Puri, Amit Dhurandhar, Tejaswini Pedapati, Karthikeyan Shanmugam, Dennis Wei, Kush R Varshney. 
        https://proceedings.neurips.cc/paper/2021/file/b538f279cb2ca36268b23f557a831508-Paper.pdf
    """


    def __init__(self, connections):
        '''
        Initialize CoFrNet_Model
        '''
        super(CoFrNet_Model, self).__init__()
        self.connections = connections  #3D array
        self.input_features = len(connections[0])
        self.output_features = len(connections[-1][0])
        self.num_total_parameters = 0
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList()

        for i in range(0, len(self.connections)):
            if i == len(self.connections) - 1: 
                self.layers.append(nn.Linear(len(self.connections[i]), self.output_features)) #connection to output layer
            else:
                self.num_total_parameters = self.num_total_parameters + np.count_nonzero(np.asarray(self.connections[i]))
                self.layers.append(CustomizedLinear(torch.tensor(self.connections[i])))
    

    def modified_reciprocal_activation(self, Wx, epsilon = .1):
        '''
        Activation function that uses capped 1/x described in paper. Takes in Wx, returns modified activation function of Wx 
        '''
        epsilonMatrix = torch.mul(torch.full_like(Wx, epsilon), torch.sign(Wx))
        denom = torch.where(torch.abs(Wx) < epsilon, epsilonMatrix, Wx)
        return torch.reciprocal(denom)

    def forward(self, x):
        '''
        Customized forward function. 
        '''
        #l_output --> layer output, a_output --> activation output
        for i in range(len(self.layers)):
            if (i == 0):
                l_input = x
                l_output = self.layers[i](l_input)
                a_output = self.modified_reciprocal_activation(l_output)
            elif ((i > 0) and (i != len(self.layers) - 1)):
                l_input = x
                l_output = self.layers[i](l_input) + prev_output
                #l_output = self.dropout(l_output)
                a_output = self.modified_reciprocal_activation(l_output)
            else:
                l_input = prev_output
                #l_input = self.dropout(l_input)
                l_output = self.layers[i](l_input)
                a_output = l_output
            prev_output = a_output
        return a_output







class CoFrNet_Explainer(DIExplainer):
    def __init__(self, cofrnet_model):
        self.model = cofrnet_model

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass
        
    def print_accuracy(self, xtest, ytest):

        results = self.model(xtest).detach().numpy()
        idx = np.argmax(results, axis = -1)
        results = np.zeros(results.shape)
        results[ np.arange(results.shape[0]), idx] = 1
        results = results.argmax(axis = -1)

        numTotal = 0
        numCorrect = 0
        for i in range(0, len(results)):
            if results[i] == ytest[i]:
                numCorrect = numCorrect + 1
            numTotal = numTotal + 1
        print("Accuracy: ", numCorrect/numTotal)
        accuracy = float(numCorrect/numTotal)


    def explain(self, explain_mode, max_layer_num = 10, var_num = 6):
        '''
        Provides Explanations of CoFrNet Model

        Args:
        explain_mode: either "importances" or "print_co_fr", will raise exception if not one of these two options
        max_layer_num: For "print_co_fr": Choose Depth of Ladder to Show, Default 10
        var_num: For "print_co_fr": Variable (index of input feature) for Which to Display Ladder, Default 6
        '''
        
        def importances(self):
            final_layer_weights = vars(self.model.layers[-1])['_parameters']['weight'].data.numpy()
            weights_by_node = final_layer_weights.T
            averaged = np.average(weights_by_node, axis = 1)
            copy_averaged = averaged.copy()
            print(copy_averaged)
            num_important_to_print = 3
            for x in range(0, num_important_to_print):
                min_idx = np.argmax(copy_averaged)
                print("The number " + str(x+1) + " most important input feature was the " + str(min_idx+1) + "th one.")
                copy_averaged[np.argmax(copy_averaged)] = copy_averaged[np.argmin(copy_averaged)]
                #print(vars(self.model.layers[-1])['_parameters']['weight'].data.numpy().T)
    

        def print_co_fr(self, max_layer_num = 10, var_num = 6): 
            #max_layer_num = chosen depth of ladder to show (10 layers, index would be 9)
            #var_num = variable for which to display ladder
            thingToPrint = ""
            for layerNum in range(0, max_layer_num-1):
                temp = vars(self.model.layers[layerNum])
                print()
                print("LayerNum: ", layerNum)
                val = (temp['_parameters']['weight'].data[var_num][var_num]).numpy()
                print("Val: ", val)
                bias = temp['_parameters']['bias'].data[var_num].numpy()
                print("Bias: ", bias)
                if (bias > (.01 * val)):
                    print(str(bias))
                    combined = "("+str(val) + "*x + " + str(bias)+")"
                    print("Combined: ", combined)
                else:
                    print("hi")
                    combined = "(" + str(val)+"*x" + "+0)"
                    print("Combined: ", combined)
                print()
                thingToPrint = "1/(" + combined + " + (" + thingToPrint + "))"

            print(thingToPrint)
            return thingToPrint

        if explain_mode == "importances": 
            importances()
        elif explain_mode == "print_co_fr": 
            print_co_fr(max_layer_num, var_num)
        else:
            raise Exception("explain_mode must be either 'importances' or 'print_co_fr'")
        
