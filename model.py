import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from collections import OrderedDict

class FCNetwork(nn.Module):
    def __init__(self, layer_units, drop_p=0.2):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            layer_units: list of integers, the sizes of all the layers
            drop_p: probability of dropout
        
        '''
        super().__init__()
        self.layer_units = layer_units.copy()
        
        try:
            output_size = layer_units.pop()
            input_size = layer_units.pop(0)
        except:
            print("RuntimeError: at least 2 values required in layer_units vector")
            return 
        
        if len(layer_units)>0:
            # Input to a hidden layer
            self.hidden_layers = nn.ModuleList([nn.Linear(input_size, layer_units[0])])

            # Add a variable number of more hidden layers
            layer_sizes = zip(layer_units[:-1], layer_units[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

            self.output = nn.Linear(layer_units[-1], output_size)
            
            self.dropout = nn.Dropout(p=drop_p)
        else:
            self.hidden_layers = nn.ModuleList()
            self.output = nn.Linear(input_size, output_size)
        
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        x = x.view(x.size(0), -1)
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return x
    
    def get_layer_units(self):
        return self.layer_units
    
    
def train_model(model, trainloader, optimizer, criterion, epochs, device='cuda', print_every=50, validationloader=None, eval_batches=None):
    ''' Trains given model with data from trainloader
    
        Arguments
        ---------
        model: model to train OR 2-tuple containing feature_model and output_model where output_model is trained
        trainloader: dataloader for training data
        optimizer: pytorch optimizer for minimizing loss 
        criterion: loss function
        epochs: number of passes over training data
        device: cpu or cuda
        print_every: number of steps to print loss after
        validationloader: dataloader for validation data
        eval_batches: number of batches to evaluate model on

        Returns
        -------
        model: trained model
    
    ''' 
    if type(model)==tuple:
        model = nn.Sequential(OrderedDict([('feature_model',model[0]),
                                      ('output_model',model[1])]))
    if type(epochs)==int:
        epochs = (0,epochs)
        
    #freeze parameters of pretrained model
    for param in model.feature_model.parameters():
        param.requires_grad = False
    
    
    model.to(device)   
    model.train()
    
    metrics = ['train_loss', 'train_accuracy', 'valid_loss', 'valid_accuracy']
    if validationloader is None:
        metrics = metrics[:2]
    print(''.join([m.ljust(25, ' ') for m in metrics]))
    scores = scores = {m:0 for m in metrics}
    
    steps = 0          
    for e in range(epochs[0],epochs[1]):
        print(f'Epoch {e}.. ')
        for inputs,labels in trainloader:
            optimizer.zero_grad()
            steps += 1
            
            inputs,labels = inputs.to(device), labels.to(device)
            
            output = model.forward(inputs)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            
            if steps%print_every==0:
                model.eval()
                        
                scores['train_loss'], scores['train_accuracy'] = evaluate_model(model, trainloader, criterion, nb_batches=eval_batches, device=device)
                
                if validationloader is not None:
                    scores['valid_loss'], scores['valid_accuracy'] = evaluate_model(model, validationloader, criterion, nb_batches=eval_batches, device=device)                         
              
                print(''.join([str(scores[m]).ljust(25, ' ') for m in metrics]))
                model.train()
                
    return model,scores  


def evaluate_model(model, dataloader, criterion, nb_batches=None, device='cuda'):
    loss = 0
    accuracy = 0
    i = 0
    model.to(device)
    model.eval()
    for inputs, labels in dataloader:
        i += 1
        
        inputs,labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        loss += criterion(output, labels).item()

        #ps = torch.exp(output)
        equality = (labels.data == output.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean().item()
        
        if nb_batches is not None:
            if i>=nb_batches:
                break
    
    n = len(dataloader) if nb_batches is None else nb_batches
    return loss/n, accuracy/n