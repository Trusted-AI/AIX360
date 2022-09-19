import unittest
import os
import shutil
import sys 

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np 
from torch.utils.data import Dataset
import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  
import torch.optim as optim

import random

#sys.path.append("../../")

import kaggle

from aix360.algorithms.cofrnet.Customized_Linear_Classes import CustomizedLinearFunction
from aix360.algorithms.cofrnet.Customized_Linear_Classes import CustomizedLinear
from aix360.algorithms.cofrnet.utils import generate_connections
from aix360.algorithms.cofrnet.utils import process_data
from aix360.algorithms.cofrnet.utils import train
from aix360.algorithms.cofrnet.utils import OnlyTabularDataset
from aix360.algorithms.cofrnet.CoFrNet import CoFrNet_Model
from aix360.algorithms.cofrnet.CoFrNet import generate_connections
from aix360.algorithms.cofrnet.CoFrNet import CoFrNet_Explainer




class TestCoFrNets(unittest.TestCase):

    def test_CoFrNet(self):

        network_depth = 13
        nput_size = 30
        output_size = 2
        cofrnet_version = "diag_ladder_of_ladder_combined"
        model = CoFrNet_Model(generate_connections(network_depth,
                                                    input_size,
                                                    output_size,
                                                    cofrnet_version))
        data = load_breast_cancer()
        X = torch.from_numpy(data['data'])
        y = torch.from_numpy(data['target'])
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        from sklearn.preprocessing import StandardScaler
        sc = MinMaxScaler(feature_range=(0,1))
        X = sc.fit_transform(X)
        X.argmax()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = 0.3,
                                                            random_state = 100,
                                                            shuffle = True)
        X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.05,
                                                            random_state=100,
                                                            shuffle = True)
        #CONVERTING TO TENSOR
        tensor_x_train = torch.Tensor(X_train)
        tensor_x_val = torch.Tensor(X_val)
        tensor_x_test = torch.Tensor(X_test)
        tensor_y_val = torch.Tensor(y_val).long()
        tensor_y_train = torch.Tensor(y_train).long()
        tensor_y_test = torch.Tensor(y_test).long()
        
        train_dataset = OnlyTabularDataset(tensor_x_train, 
                                            tensor_y_train)

        batch_size = 100
        dataloader = DataLoader(train_dataset, batch_size) 


        train(model, dataloader, output_size)

        explainer = CoFrNet_Explainer(model)
        explainer.print_accuracy(tensor_x_test, y_test)

if __name__ == '__main__':
    unittest.main()