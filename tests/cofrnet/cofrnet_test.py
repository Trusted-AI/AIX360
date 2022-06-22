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

sys.path.append('../../')

from aix360.algorithms.cofrnet.Customized_Linear_Classes import CustomizedLinearFunction
from aix360.algorithms.cofrnet.Customized_Linear_Classes import CustomizedLinear
from aix360.algorithms.cofrnet.utils import generate_connections
from aix360.algorithms.cofrnet.utils import process_data
from aix360.algorithms.cofrnet.CoFrNet import CoFrNet_Model
from aix360.algorithms.cofrnet.CoFrNet import generate_connections
from aix360.algorithms.cofrnet.CoFrNet import CoFrNet_Explainer



class TestCoFrNets(unittest.TestCase):

    def test_CoFrNet(self):

        seed_num = 158
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        np.random.seed(seed_num)


        network_depth = 13
        input_size = 40
        output_size = 3
        cofrnet_version = "diag_ladder_of_ladder_combined"
        #Create CoFrNet
        model = CoFrNet_Model(generate_connections(network_depth, 
                                                    input_size, 
                                                    output_size, 
                                                    cofrnet_version))







        #Get Data
        tensor_x_train, tensor_y_train, tensor_x_val, tensor_y_val, tensor_x_test, y_test = process_data(data_filename= 'waveformnoise.csv', 
                                                                                                        first_column_csv = 0, 
                                                                                                        last_column_csv = -1)
        class OnlyTabularDataset(Dataset):
            def __init__(self, values, label):
                self.values = values
                self.label = label

            def __len__(self):
                return len(self.label)

            def __getitem__(self, index):
                return {
                    'tabular': torch.tensor(self.values[index], dtype=torch.float),
                    'target' :  torch.tensor(self.label[index], dtype=torch.long)
                    }

        train_dataset = OnlyTabularDataset(tensor_x_train, 
                                            tensor_y_train)

        dataloader = DataLoader(train_dataset, 40)




        #Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        EPOCHS = 20
        for epoch in range(EPOCHS):  # loop over the dataset multiple times

            running_loss = 0.0
            #for i, data in enumerate(trainloader, 0):
            for i, batch in tqdm(enumerate(dataloader)):
                # get the inputs; data is a list of [inputs, labels]
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(batch['tabular'])
                loss = criterion(outputs, batch['target'])
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')

        print('Finished Training')





        explainer = CoFrNet_Explainer(model)
        explainer.print_accuracy(tensor_x_test, y_test)
        explainer.importances()
        explainer.print_co_fr()

if __name__ == '__main__':
    unittest.main()