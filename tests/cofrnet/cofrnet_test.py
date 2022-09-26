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
from sklearn.datasets import load_breast_cancer, load_wine, load_linnerud, load_diabetes, load_iris, load_digits
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import random
import pandas as pd


from aix360.algorithms.cofrnet.CustomizedLinearClasses import CustomizedLinearFunction
from aix360.algorithms.cofrnet.CustomizedLinearClasses import CustomizedLinear
from aix360.algorithms.cofrnet.utils import generate_connections
from aix360.algorithms.cofrnet.utils import process_data
from aix360.algorithms.cofrnet.utils import train
from aix360.algorithms.cofrnet.utils import OnlyTabularDataset
from aix360.algorithms.cofrnet.CoFrNet import CoFrNet_Model
from aix360.algorithms.cofrnet.CoFrNet import CoFrNet_Explainer





class TestCoFrNets(unittest.TestCase):

    def test_CoFrNet(self):

        network_depth = 13
        input_size = 40
        output_size = 3
        cofrnet_version = "diag_ladder_of_ladder_combined"
        model = CoFrNet_Model(generate_connections(network_depth,
                                                    input_size,
                                                    output_size,
                                                    cofrnet_version))


        first_column_csv = 0
        last_column_csv = -1


        web_link = 'http://www.dropbox.com/s/qtdv1teptf097zl/waveformnoise.csv?dl=1'
        tensor_x_train, tensor_y_train, tensor_x_val, tensor_y_val, tensor_x_test, y_test = process_data(first_column_csv = first_column_csv, 
                                                                                                            last_column_csv = last_column_csv, 
                                                                                                            web_link=web_link)
        
        train_dataset = OnlyTabularDataset(tensor_x_train, 
                                            tensor_y_train)

        batch_size = 100
        dataloader = DataLoader(train_dataset, batch_size) 


        train(model, dataloader, output_size)

        explainer = CoFrNet_Explainer(model)
        explainer.print_accuracy(tensor_x_test, y_test)

if __name__ == '__main__':
    unittest.main()