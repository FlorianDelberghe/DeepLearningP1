# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import os
from torch.nn import functional as F

import torchvision
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
import xgboost as xgb
from Proj1 import progress_bar, compare_digits, create_Net
from Nets import *
import pickle

class MLP(nn.Module):
    """
    """
    def __init__(self, param):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(20, param['hidden'][0])
        self.fc2 = nn.Linear(param['hidden'][0], param['hidden'][1])
        self.fc3 = nn.Linear(param['hidden'][1], 2)
        
        self.naive = param['naive']
        self.fcnaive = nn.Linear(2, 2)
        
        if param['activation'] == 'tanh':
            self.activation = F.tanh
        elif param['activation'] == 'relu':
            self.activation = F.relu
        else :
            raise ValueError("Activation must be 'relu' or 'tanh'")
        
        if param['norm'] == 'batch':
            self.norm1 = nn.BatchNorm1d(20)
            self.norm2 = nn.BatchNorm1d(param['hidden'][0])
            self.norm3 = nn.BatchNorm1d(param['hidden'][1])
        elif param['norm'] == 'dropout':
            self.norm1 = nn.Dropout(param['drop_proba'][0])
            self.norm2 = nn.Dropout(param['drop_proba'][1])
            self.norm3 = nn.Dropout(param['drop_proba'][2])
        else :
            raise ValueError("Normalisation must be 'batch' or 'dropout'")        
        
    def forward(self, input_):
        
        if self.naive:
            x = Variable(torch.stack([input_[:,:10].data.max(1)[1].float(), input_[:,10:].data.max(1)[1].float()], 1))
            x = self.fcnaive(x)
        else:
            x = Variable(input_)
            x = self.activation(self.fc1(self.norm1(x)))
            x = self.activation(self.fc2(self.norm2(x)))
            x = self.fc3(self.norm3(x))
        
        return x
    

def pickle_output(param):
    """
    """    
    train_input, train_target, train_class, test_input, test_target, test_class = \
        prologue.generate_pair_sets(N_PAIRS)
        
    # Normalization of the data
    train_input /= train_input.max()
    test_input /= test_input.max()
    
    model = create_Net(param)
    model.load_state_dict(torch.load("nets/{}.pkl".format(param['net'])))
    
    x_out_tr, y_out_tr, _ = model.eval()(train_input)
    x_out_te, y_out_te, _ = model.eval()(test_input)
    class_proba_tr, class_proba_te = torch.cat([x_out_tr, y_out_tr], 1), torch.cat([x_out_te, y_out_te], 1)
    
    pickle.dump({'tr_pred_proba': class_proba_tr, 'tr_target': train_target, 'tr_class': train_class,
                 'te_pred_proba': class_proba_te, 'te_target': test_target, 'te_class': test_class}, 
                open("data/{}_pred.pkl".format(param['net']), 'wb'))    
    

def train_model(model, train_input, train_target, mini_batch_size, nb_epoch=40, crit='cross_entropy', opt='SGD',  progress=True):
    """
    """
    if crit == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
        
    if opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=1e-1)
    else:
        raise NotImplementedError
    
    # Training of the classification net
    for i in range(nb_epoch):
        for b in range(0, train_input.size(0), mini_batch_size):
            pred_target = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(pred_target, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step() 
        if progress:
            print("\r{} loss={:.6E} {}".format(crit.capitalize(), loss.item(), progress_bar(i+1, nb_epoch)), end='')    
            

def test_model(model, test_input, test_target):
    """
    Evaluates teh model with classification and comparison
    """
    model.train(False)
    
    pred_target = model(test_input)
    comp_err = (pred_target.data.max(1)[1].numpy() != test_target.numpy()).sum() 
    
    print("Net comparison error: {:.4}%, {}/{}".format(comp_err /test_target.shape[0] *100, comp_err, test_target.shape[0]))    

    return comp_err, comp_err /test_target.size(0) *100
    
    
def __main__():
    cnet2 = {"net": 'compNet2', "hidden": 120, "epochs": 80, "batch_size": 10, 
             "pool": 'max', "activation": 'relu', "drop_proba": [0.05, 0.05, 0.5, 0.2], "seed": None}
    cnet4 = {"net": 'compNet4', "hidden": 350, "epochs": 30, "batch_size": 10, 
             "pool": 'max', "activation": 'tanh', "drop_proba": [0, 0, 0, 0.2, 0, 0, 0], "seed": None}
    
    pickle_output(cnet2)
    pickle_output(cnet4)
    
    data = pickle.load(open("data/{}_pred.pkl".format('compnet4'), 'rb'))
    train_input, test_input = data['tr_pred_proba'], data['te_pred_proba']
    train_target, test_target = data['tr_target'], data['te_target']
    
    mlp = {"net": 'MLP', "hidden": [60, 90], "epochs": 70, "batch_size": 100, 
           "activation": 'relu', "norm": 'batch', "drop_proba": [0., 0., 0.6], "naive": False,  "seed": None}    
    
    model = MLP(mlp)
    train_model(model, train_input, train_target, mini_batch_size=mlp['batch_size'], 
                nb_epoch=mlp['epochs'], crit='cross_entropy', opt='SGD')
    test_model(model, test_input, test_target)
    
    
if __name__ == '__main__':
    __main__()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    