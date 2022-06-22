'''
author: @ishapuri101
'''

import numpy as np # linear algebra
from sklearn.preprocessing import MinMaxScaler
import torch # import main library
import pandas as pd #loading data in table form  


def generate_connections(num_total_layers: int, input_size: int, output_size: int, which_variant: str, num_nodes = 0): 
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

    def diagonalized_ladder_of_ladders_combinedd():
        #numLayers DOES include output layer

        #numLayers = numLadders in this case

        getConns = []
        numLayers_notIncOutput = num_total_layers - 1 #numLayers_notIncOutput = input_size
        
        for i in range(0, numLayers_notIncOutput):
            ladderOfLadders = np.ones([input_size, numLayers_notIncOutput])
            ladderOfLadders[:, 0:i] = 0 
            toAppend = np.append(np.eye(input_size), ladderOfLadders, axis = 1)
            getConns.append(toAppend.tolist())
        #print(len(getConns[-1][0]))
        getConns.append(np.ones([len(getConns[-1][0]), output_size]).tolist())
        #getConns.append(np.ones([numLayers_notIncOutput, output_size]).tolist())

        return getConns

    
    if which_variant == "fully_connected":
        return fully_connected_constant()
    elif which_variant == "diagonalized":
        return diagonalized()
    elif which_variant == "ladder_of_ladders":
        return ladder_of_ladders()
    elif which_variant == "diag_ladder_of_ladder_combined":
        return diagonalized_ladder_of_ladders_combinedd()
    else:
        raise Exception("You must choose one of the following four choices for which_variant: fully_connected, diagonalized, ladder_of_ladders, or diag_ladder_of_ladder_combined")
                







def process_data(data_filename, first_column_csv, last_column_csv):
    #data_filename: filename of data source
    #first_column_csv: index (starting from 0) of first column to include in dataset
    #last_column_csv: index (starting from 0) of last column to include in dataset. Use -1 if you want to include all of the columns. 

    import pandas as pd
    df=pd.read_csv(data_filename, sep=',',header=0)   
    if last_column_csv != -1: 
        last_column_csv = last_column_csv + 1
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=seed)

    #CONVERTING TO TENSOR
    tensor_x_train = torch.Tensor(X_train)
    tensor_x_val = torch.Tensor(X_val)
    tensor_x_test = torch.Tensor(X_test)

    tensor_y_val = torch.Tensor(y_val).long()
    tensor_y_train = torch.Tensor(y_train).long()
    tensor_y_test = torch.Tensor(y_test).long()

    return tensor_x_train, tensor_y_train, tensor_x_val, tensor_y_val, tensor_x_test, y_test


