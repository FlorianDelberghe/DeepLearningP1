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

N_PAIRS = 1000    
    
def train_model(model, train_input, train_target, train_class, mini_batch_size, nb_epoch=40, crit='cross_entropy', opt='SGD',  progress=True):
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
            out_x, out_y, _ = model(train_input.narrow(0, b, mini_batch_size))
            loss_x = criterion(out_x, train_class[:,0].narrow(0, b, mini_batch_size))
            loss_y = criterion(out_y, train_class[:,1].narrow(0, b, mini_batch_size))
            model.zero_grad()
            (loss_x + loss_y).backward()
            optimizer.step() 
        if progress:
            print("\r{} loss={:.6E} {}".format(crit.capitalize(), (loss_x + loss_y).item(), progress_bar(i+1, nb_epoch)), end='')
    
    # Training of the comparison net      
    for i in range(nb_epoch):
        for b in range(0, train_input.size(0), mini_batch_size):
            # forward in eval() mode to discard the dropout 
            _, _, out_z = model.eval()(train_input.narrow(0, b, mini_batch_size))
            loss_z = criterion(out_z, train_target.narrow(0, b, mini_batch_size))       
            model.zero_grad()
            (loss_z).backward()
            optimizer.step() 
        if progress:
            print("\r{} loss={:.6E} {}".format(crit.capitalize(), (loss_z).item(), progress_bar(i+1, nb_epoch)), end='')


def test_model(model, test_input, test_target, test_class):
    """
    Evaluates teh model with classification and comparison
    """
    model.train(False)
    
    pred_x, pred_y, pred_z = model(test_input)
    class_err = (pred_x.data.max(1)[1].numpy() != test_class[:,0].numpy()).sum()
    class_err += (pred_y.data.max(1)[1].numpy() != test_class[:,1].numpy()).sum()
    comp_err = (pred_z.data.max(1)[1].numpy() != test_target.numpy()).sum()        

    pred_class = torch.stack([pred_x.data.max(1)[1], pred_y.data.max(1)[1]], dim=1)
    print("Naive comparison error: {}".format((compare_digits(pred_class) - test_target.int()).abs().sum().item()))
            
    return class_err, class_err /test_target.size(0) /2*100, comp_err, comp_err /test_target.size(0) *100


def test_param(param):
    
    print("Testing...", end='')
    for k in param.keys():
        print(" {}: {} |".format(k, param[k]), end='')
    print('')    
    
    train_input, train_target, train_class, test_input, test_target, test_class = \
        prologue.generate_pair_sets(N_PAIRS)
        
    # Normalization of the data
    train_input /= train_input.max()
    test_input /= test_input.max()
    
    # setting a random seed
    if param['seed'] != None:
        torch.manual_seed(param['seed'])
        
    model = create_Net(param)  
    model.train(True)
    train_model(model, train_input, train_target, train_class, param['batch_size'], param['epochs'])    
    model.train(False)
    
    class_err, class_per, comp_err, comp_per = test_model(model,  test_input, test_target, test_class)
    
    print('Classification test error: {:0.2f}% {:d}/{:d}'.format(class_per, class_err, 2*N_PAIRS))
    
    print("Net comparison error: {:0.2f}% {:d}/{:d}".format(comp_per, comp_err, N_PAIRS))


def __main__():
    torch.manual_seed(999)
    
    cnet2 = {"net": 'compNet2', "hidden": 120, "epochs": 80, "batch_size": 10, 
             "pool": 'max', "activation": 'relu', "drop_proba": [0.05, 0.05, 0.5, 0.2], "seed": None}
    cnet4 = {"net": 'compNet4', "hidden": 350, "epochs": 30, "batch_size": 10, 
             "pool": 'max', "activation": 'tanh', "drop_proba": [0, 0, 0, 0.1], "seed": None}
    
    test_param(cnet4)


if __name__ == "__main__":
    __main__()













































