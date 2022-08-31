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
sys.path.append("AIX360/aix360/algorithms/cofrnet")

from aix360.algorithms.cofrnet.Customized_Linear_Classes import CustomizedLinearFunction
from aix360.algorithms.cofrnet.Customized_Linear_Classes import CustomizedLinear
from aix360.algorithms.cofrnet.utils import generate_connections
from aix360.algorithms.cofrnet.utils import process_data
from aix360.algorithms.cofrnet.utils import train
from aix360.algorithms.cofrnet.utils import OnlyTabularDataset
from aix360.algorithms.cofrnet.CoFrNet import CoFrNet_Model
from aix360.algorithms.cofrnet.CoFrNet import generate_connections
from aix360.algorithms.cofrnet.CoFrNet import CoFrNet_Explainer
import kaggle

os.environ['KAGGLE_USERNAME'] = 'ishaopensourceibm' #replace with your Kaggle username
os.environ['KAGGLE_KEY'] = 'e38322b9c75dc4b64d7198d7c43a598c' #replace with your Kaggle api key


class TestCoFrNets(unittest.TestCase):

    def test_CoFrNet(self):

        network_depth = 13
        input_size = 10
        output_size = 2
        cofrnet_version = "diag_ladder_of_ladder_combined"

        model = CoFrNet_Model(generate_connections(network_depth, 
                                                    input_size, 
                                                    output_size, 
                                                    cofrnet_version))

        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files('abhinand05/magic-gamma-telescope-dataset', path=".", unzip = True)



        data_filename= 'telescope_data.csv'
        first_column_csv = 1
        last_column_csv = -1


        tensor_x_train, tensor_y_train, tensor_x_val, tensor_y_val, tensor_x_test, y_test = process_data(data_filename= data_filename, 
                                                                                                first_column_csv = first_column_csv, 
                                                                                                last_column_csv = last_column_csv)

        train_dataset = OnlyTabularDataset(tensor_x_train, 
                                            tensor_y_train)

        batch_size = 100
        dataloader = DataLoader(train_dataset, batch_size) 


        train(model, dataloader, output_size)

        explainer = CoFrNet_Explainer(model)
        explainer.print_accuracy(tensor_x_test, y_test)

if __name__ == '__main__':
    unittest.main()