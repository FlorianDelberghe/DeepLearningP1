# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import optim
from torch import nn
import os

import torchvision
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
from Utilities import progress_bar, compare_digits
from CompNets import *
    
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
            # forward in eval() mode to discard the dropout from the conv part of the net
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
    
    # The data is small, can be done all at once
    pred_x, pred_y, pred_z = model(test_input)
    class_err = (pred_x.data.max(1)[1].numpy() != test_class[:,0].numpy()).sum()
    class_err += (pred_y.data.max(1)[1].numpy() != test_class[:,1].numpy()).sum()
    comp_err = (pred_z.data.max(1)[1].numpy() != test_target.numpy()).sum()        
    
    # Uses naive comparison for when the is not in naive mode
    pred_class = torch.stack([pred_x.data.max(1)[1], pred_y.data.max(1)[1]], dim=1)
    print("Naive comparison error: {}".format((compare_digits(pred_class) - test_target.int()).abs().sum().item()))
            
    return class_err, class_err /test_target.size(0) /2*100, comp_err, comp_err /test_target.size(0) *100


def test_param(param, save=False, log=False):
    
    print("Testing...", end='')
    for k in param.keys():
        print(" {}: {} |".format(k, param[k]), end='')
    print('')    
    
    N_PAIRS = 1000
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
    
    if log:
        with open("{}results.log".format(param['net']+'fc'), mode='at') as f:
            f.write("Class error: {:.4}%, Comp error: {:.4}%\n".format(class_per, comp_per))
            
    if save:
        if 'nets' not in os.listdir():
                dir_ = 'nets'
                try:  
                    os.mkdir(dir_)
                except OSError:  
                    print ("Creation of the directory %s failed" % dir_)
                else:  
                    print ("Successfully created the directory %s " % dir_)
                        
        torch.save(model.state_dict(), "nets/{}.pkl".format(param['net']))




































