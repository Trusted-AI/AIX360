'''
author: @ishapuri101
'''

import numpy as np # linear algebra
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd #loading data in table form  
from torch.utils.data import Dataset
import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  
import torch.optim as optim
#import random
from torch.utils.data import DataLoader




def generate_connections(num_total_layers: int, 
                        input_size: int, 
                        output_size: int, 
                        which_variant: str, 
                        num_nodes = 0,
                        feature_index = 0, 
                        features_to_use = []): 
    def fully_connected_constant(): 
        num_total_layers = num_total_layers - 1 #not including output layer

        genConns = []

        for i in range(0, num_total_layers):
            genConns.append(np.ones([input_size, num_nodes]).tolist())

        genConns.append(np.ones([num_nodes, output_size]).tolist())
        return genConns

    def diagonalized(): 
        #numLayers DOES include output layer

        numLayers_notInclOutput = num_total_layers - 1
        genConns = []

        for i in range(0, numLayers_notInclOutput):
            genConns.append(np.eye(input_size).tolist())

        genConns.append(np.ones([input_size, output_size]).tolist())
        return genConns

    def ladder_of_ladders():
        getConns = []
        numLayers_notIncOutput = num_total_layers - 1 #numLayers_notIncOutput = input_size

        for i in range(0, numLayers_notIncOutput):
            toAppend = np.ones([input_size, numLayers_notIncOutput])
            toAppend[:, 0:i] = 0 #toAppend[:, input_size-i:input_size] = 0
            getConns.append(toAppend.tolist())#i added this tolist() on april 20th

        getConns.append(np.ones([numLayers_notIncOutput, output_size]).tolist())

        return getConns

    def diagonalized_ladder_of_ladders_combined():
        #numLayers DOES include output layer

        #numLayers = numLadders in this case

        getConns = []
        numLayers_notIncOutput = num_total_layers - 1 #numLayers_notIncOutput = input_size
        
        for i in range(0, numLayers_notIncOutput):
            ladderOfLadders = np.ones([input_size, numLayers_notIncOutput])
            ladderOfLadders[:, 0:i] = 0 
            toAppend = np.append(np.eye(input_size), ladderOfLadders, axis = 1)
            getConns.append(toAppend.tolist())
        getConns.append(np.ones([len(getConns[-1][0]), output_size]).tolist())

        return getConns

    def one_feature_diagonalized():
        #numLayers DOES include output layer

        numLayers_notInclOutput = num_total_layers - 1
        genConns = []

        for i in range(0, numLayers_notInclOutput):
            toAppend = np.zeros([input_size, 1])
            toAppend[feature_index][0] = 1
            genConns.append(toAppend.tolist())

        genConns.append(np.ones([1, output_size]).tolist())

        return genConns

    def n_feature_fully_connected():
        #numLayers DOES include output layer

        numLayers_notInclOutput = num_total_layers - 1
        features_not_to_use = []
        for i in range(0, input_size):
            if i not in features_to_use:
                features_not_to_use.append(i)
        genConns = []

        for i in range(0, numLayers_notInclOutput):
            toAppend = np.ones([input_size, input_size])
            for num in features_not_to_use:
                for j in range(0, input_size):
                    toAppend[j][num] = 0
                    toAppend[num][j] = 0
            genConns.append(toAppend.tolist())

        genConns.append(np.ones([input_size, output_size]).tolist())

        return genConns

    
    if which_variant == "fully_connected":
        return fully_connected_constant()
    elif which_variant == "diagonalized":
        return diagonalized()
    elif which_variant == "ladder_of_ladders":
        return ladder_of_ladders()
    elif which_variant == "diag_ladder_of_ladder_combined":
        return diagonalized_ladder_of_ladders_combined()
    elif which_variant == "one_feature_diag":
        return one_feature_diagonalized()
    elif which_variant == 'n_feature_fully_connected':
        return n_feature_fully_connected()
    else:
        raise Exception("You must choose one of the following four choices for which_variant: fully_connected, diagonalized, ladder_of_ladders, diag_ladder_of_ladder_combined, or one_feature_diag")
                







def process_data(first_column_csv, last_column_csv, web_link = None, data_filename = None):
    #data_filename: filename of data source
    #first_column_csv: index (starting from 0) of first column to include in dataset
    #last_column_csv: index (starting from 0) of last column to include in dataset. Use -1 if you want to include all of the columns. 

    import pandas as pd

    if web_link is not None: 
        pathname = web_link
    else:
        pathname = data_filename #'datasets/' + data_filename
    df=pd.read_csv(pathname, sep=',',header=0, lineterminator='\r')   
    if last_column_csv != -1: 
        last_column_csv = last_column_csv + 1
    df = df.sample(frac = 1)
    X = df.iloc[:, first_column_csv : last_column_csv].values 
    
    y = df.iloc[:,-1].values.T




    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    from sklearn.preprocessing import StandardScaler
    sc = MinMaxScaler(feature_range=(0,1))
    X = sc.fit_transform(X)
    X.argmax()

    seeds = [1, 10, 100, 555, 9897]
    seed = seeds[2]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.3, 
                                                        random_state = seed, 
                                                        shuffle = True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size=0.05, 
                                                        random_state=seed,
                                                        shuffle = True)

    #CONVERTING TO TENSOR
    tensor_x_train = torch.Tensor(X_train)
    tensor_x_val = torch.Tensor(X_val)
    tensor_x_test = torch.Tensor(X_test)

    tensor_y_val = torch.Tensor(y_val).long()
    tensor_y_train = torch.Tensor(y_train).long()
    tensor_y_test = torch.Tensor(y_test).long()

    return tensor_x_train, tensor_y_train, tensor_x_val, tensor_y_val, tensor_x_test, y_test


def onehot_encoding(label, n_classes):
    """Conduct one-hot encoding on a label vector."""
    label = label.view(-1)
    onehot = torch.zeros(label.size(0), n_classes).float().to(label.device)
    onehot.scatter_(1, label.view(-1, 1), 1)
   
    return onehot


    





def train(model, dataloader, num_classes, lr = 0.001, momentum = 0.9, epochs = 20):
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss(reduction="sum")
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    EPOCHS = epochs
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        print("Epoch: ", epoch)
        running_loss = 0.0
        #for i, data in enumerate(trainloader, 0):
        for i, batch in tqdm(enumerate(dataloader)):
            # get the inputs; data is a list of [inputs, labels]
            # forward + backward + optimize

            batch['tabular'].requires_grad=True


            outputs = model(batch['tabular'])

            one_hot_encoded_target = onehot_encoding(batch['target'], num_classes)
            
            #loss = criterion(outputs, batch['target'])
            loss = criterion(outputs, one_hot_encoded_target)

            # zero the parameter gradients
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        print("Loss: ", running_loss)

    print('Finished Training')

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