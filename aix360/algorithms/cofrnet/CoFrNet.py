'''
author: @ishapuri101, @sadhamanus
'''

import pandas as pd #loading data in table form  
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torchsample
import os
from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from Customized_Linear_Classes import CustomizedLinearFunction
from Customized_Linear_Classes import CustomizedLinear
from utils import generate_connections
from utils import process_data

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

from torch.utils.data import DataLoader

from tqdm import tqdm

from torch.utils.data import Dataset


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
        
        
        self.BatchNorm = nn.BatchNorm2d(self.input_features)

    def modified_reciprocal_activation(self, Wx, epsilon = .01):
        '''
        Activation function that uses capped 1/x described in paper. Takes in Wx, returns modified activation function of Wx 
        '''
        epsilonMatrix = torch.mul(torch.full_like(Wx, epsilon), torch.sign(Wx))
        denom = torch.where(torch.abs(Wx) < epsilon, epsilonMatrix, Wx)
        denom = torch.where(denom == 0, epsilon, denom)
        #print(epsilonMatrix, Wx)
        if torch.any(torch.isnan(torch.reciprocal(denom))):
            print(f'nans present when doing 1/x')
        return torch.reciprocal(denom)

       
        

    def forward(self, x):
        '''
        Customized forward function. 
        '''

         
        #l_output --> layer output, a_output --> activation output
        for i in range(len(self.layers)):
            if (i == 0):
                #print(f'input size: {x.size()}')
                l_input = x #self.BatchNorm(x) #x
                l_output = self.layers[i](l_input)
                a_output = self.modified_reciprocal_activation(l_output)
                #print("self.layers[i].output_features", self.layers[i].output_features)
                batchNorm = nn.BatchNorm1d(self.layers[i].output_features)
                a_output = batchNorm(a_output)
            elif ((i > 0) and (i != len(self.layers) - 1)):
                l_input = x #self.BatchNorm(x) #x
                l_output = self.layers[i](l_input) + prev_output
                #l_output = self.dropout(l_output)
                a_output = self.modified_reciprocal_activation(l_output)
                #batchNorm = nn.BatchNorm1d(self.layers[i].output_features)
                #a_output = batchNorm(a_output)
            else:
                l_input = prev_output
                #l_input = self.dropout(l_input)
                l_output = self.layers[i](l_input)
                a_output = l_output
            prev_output = a_output
        #print(f'output size: {a_output.size()}')
        return a_output




ckpt_path = 'ckpt/cofrnet.pt'


class CoFrNet_Explainer():
    def __init__(self, num_layers, data_input_size, data_output_size, which_variant, tensor_x_train, tensor_y_train, tensor_x_val, tensor_y_val, tensor_x_test, y_test,num_nodes):
        self.num_layers = num_layers
        self.data_input_size = data_input_size
        self.data_output_size = data_output_size
        self.which_variant = which_variant
        self.model = CoFrNet_Model(generate_connections(self.num_layers, 
                                                        self.data_input_size, 
                                                        self.data_output_size, 
                                                        self.which_variant,num_nodes))
        self.tensor_x_train = tensor_x_train
        self.tensor_y_train = tensor_y_train
        self.tensor_x_val = tensor_x_val
        self.tensor_y_val = tensor_y_val
        self.tensor_x_test = tensor_x_test
        self.y_test = y_test

        self.train_dataset = OnlyTabularDataset(self.tensor_x_train, 
                                                self.tensor_y_train)


        def collate_fn(batch):
            batch = torch.cat([sample[0].unsqueeze(0) for sample in batch], dim=0)
            return batch

        self.dataloader = DataLoader(self.train_dataset, data_input_size)
        
        self.x_train_dl = DataLoader(tensor_x_train, data_input_size)
        self.y_train_dl = DataLoader(tensor_y_train, data_input_size)
        self.x_val_dl = DataLoader(tensor_x_val, data_input_size)
        self.y_val_dl = DataLoader(tensor_y_val, data_input_size)
        self.x_test_dl = DataLoader(tensor_x_val, data_input_size)
        self.y_test_dl = DataLoader(y_test, data_input_size)

        
    def evaluate(model, dataloader):
        model.eval()

        val_accuracy = []
        val_loss = []
        
        for batch in dataloader:
        
            with torch.no_grad():
                logits = model(batch)
            loss = loss_fn(logits, target)
            val_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == target).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy


    
    def fit(self, weight_decay = 0, patience = float('Inf'), min_delta = .0001, learning_rate = 1e-2, num_epoch = 100):
       
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        EPOCHS = num_epoch
        optm = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        last_loss = float('Inf')
        min_loss = float('Inf')
        
        trigger_times = 0
        


        for epoch in range(EPOCHS):
            epoch_loss = 0
            correct = 0
            for bidx, batch in tqdm(enumerate(self.dataloader)):
                x_train = self.x_train_dl 
                y_train = self.y_train_dl 
                #print(f'batch size: {len(batch)}')
                loss, predictions = self.train(self.model,
                                                batch['tabular'],
                                                batch['target'], 
                                                optm, 
                                                criterion)
                if loss > last_loss:
                    trigger_times += 1

                
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    if not os.path.exists(ckpt_path):
                        raise RuntimeError(f'\'{ckpt_path}\' does not exist')
                    self.model.load_state_dict(torch.load(ckpt_path))
                    return self.model                
                else:
                    trigger_times = 0

                last_loss = loss

                #self.model.eval()
                for idx, i in enumerate(predictions):
                    predictions_max = torch.max(i)
                    index_of_max = list(i).index(max(list(i)))

                    if index_of_max == self.tensor_y_train[idx]:
                        correct += 1
               
                acc = (correct/len(self.y_train_dl))
                epoch_loss += loss
            
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                torch.save(self.model.state_dict(), ckpt_path)

            print('Epoch {} Accuracy : {}'.format(epoch+1, acc*100))
            print('Current best loss: {}'.format(min_loss))
            print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))


    def train(self, model, x, y, optimizer, criterion):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output,y)
        loss.backward()
        optimizer.step()


        return loss, output
        

    def predict(self, input_x_tensor):
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f'\'{ckpt_path}\' does not exist')
        self.model.load_state_dict(torch.load(ckpt_path))
        return(self.model(input_x_tensor))

    def print_accuracy(self):
        results = self.predict(self.tensor_x_test).detach().numpy()
        idx = np.argmax(results, axis = -1)
        results = np.zeros(results.shape)
        results[ np.arange(results.shape[0]), idx] = 1
        results = results.argmax(axis = -1)

        numTotal = 0
        numCorrect = 0
        for i in range(0, len(results)):
            if results[i] == self.y_test[i]:
                numCorrect = numCorrect + 1
            numTotal = numTotal + 1
        print("Accuracy: ", numCorrect/numTotal)
        accuracy = float(numCorrect/numTotal)

    def importances(self):
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f'\'{ckpt_path}\' does not exist')
        self.model.load_state_dict(torch.load(ckpt_path))
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
    

    def explain(self, max_layer_num = 10, var_num = 6): 
        #max_layer_num = chosen depth of ladder to show (10 layers, index would be 9)
        #var_num = variable for which to display ladder
        thingToPrint = ""
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f'\'{ckpt_path}\' does not exist')
        self.model.load_state_dict(torch.load(ckpt_path))
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
                #thingToPrint = "\n 1/("+str(val) + "x + " + str(bias)+")" + thingToPrint
            else:
                print("hi")
                combined = "(" + str(val)+"*x" + "+0)"
                print("Combined: ", combined)
                #thingToPrint = "\n 1/(" + str(val)+"x" + "+0)" + thingToPrint
            print()
            thingToPrint = "1/(" + combined + " + (" + thingToPrint + "))"

        print(thingToPrint)
        return thingToPrint
