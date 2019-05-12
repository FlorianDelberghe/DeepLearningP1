# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import os

import torchvision
from Nets import *
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
#import xgboost as xgb

N_PAIRS = 1000

def progress_bar(pos, total, length=50):
    """
    Retruns a string with the progression of the algorithm
    """
    if pos == total:
        return '[{}]\n'.format('#'*length)
                 
    rel_pos = int(pos/total*length)
    done = '#'*(rel_pos)
    todo = '-'*(length-rel_pos)   
    return '[{}]'.format(done+todo)

 
def compute_nb_errors(model, input_, target, mini_batch_size):
    """
    Computes the number of errors between a target vector and the model prediction using input
    """
    nb_errors = 0

    for b in range(0, input_.size(0), mini_batch_size):
        output = model(input_.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(mini_batch_size):
            if target.data[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors

       
def compare_digits(classes):
    """
    Returns the index of the largest digit
    """
    diff = ( classes[:, 1] - classes[:, 0] ).sign()
    
    # gets 0 diff to 1 classification 
    return ( diff - diff.abs() ).div(2).add(1).int()


# ================================================================================ #
# ===== The following function were used in previous builds of the project ======= #
# === They are now deprecated but were useful in the developement of some nets === #
# ================================================================================ #
    
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
    
    for i in range(nb_epoch):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))     
            model.zero_grad()
            loss.backward()
            optimizer.step() 
        if progress:
            print("\r{} loss={:.6E} {}".format(crit.capitalize(), loss.item(), progress_bar(i+1, nb_epoch)), end='')


def test_model(model, test_input, test_target, test_class):
    """
    Evaluates teh model with classification and comparison
    """
    model.train(False)
    class_err = 0               
    for i in [0, 1]:
        class_err += compute_nb_errors(model, Variable(test_input[:, i, :, :].view(-1, 1, 14, 14)), 
                                              Variable(test_class[:, i].long()), 20)  
        
    pred_class = torch.stack([model(Variable(test_input[:, 0, :, :].view(-1, 1, 14, 14))).data.max(1)[1], 
                              model(Variable(test_input[:, 1, :, :].view(-1, 1, 14, 14))).data.max(1)[1]], dim=1)
    comp_err = (compare_digits(pred_class) - test_target.int()).abs().sum().item()
                
    return class_err, class_err/test_input.size(0)/2, comp_err, comp_err/test_input.size(0)


def grid_search(param, optimize='epoch', rg=[10, 100], step=10, level=1):
    """
    Does a grid search on a Net optimising for a given parameters
    """
    def get_range_param(rg, step):
        return [i*step+rg[0] for i in range(int((rg[1]-rg[0]) / step)+1)]
    
    def train_with_set_params(model, nb_epochs, mini_batch_size):          
        model.train(True)
        for i in [0, 1]:                
                train_model(model, Variable(train_input[:, i, :, :].view(-1, 1, 14, 14)), 
                            Variable(train_class[:, i].long()), mini_batch_size, nb_epochs, progress=False)
        model.train(False)
        
    def save_figure(x_plot, y_plot, legend, ax_labels, title, filename): 
        plt.figure(figsize=(5, 3))
        for i, y in enumerate(y_plot):
            plt.plot(x_plot, y, label=legend[i])            
        plt.legend()
        plt.xlabel(ax_labels[0]); plt.ylabel(ax_labels[1]);
        
        if 'figs' not in os.listdir():
            dir_ = 'figs'
            try:  
                os.mkdir(dir_)
            except OSError:  
                print ("Creation of the directory %s failed" % dir_)
            else:  
                print ("Successfully created the directory %s " % dir_)
                
        plt.savefig(os.path.join('figs', filename))
        plt.show()   
     
        
    # Imports the data  
    train_input, train_target, train_class, test_input, test_target, test_class = \
        prologue.generate_pair_sets(N_PAIRS)
    # Normalization of the data
    train_input /= train_input.max()
    test_input /= test_input.max()  
    
    # Sets default
    nb_hidden, nb_epochs, mini_batch_size = param['hidden'], param['epochs'], param['batch_size']
    
    print("Grid search, Net: {}, optimising: {}, {}->{} step: {}".format(param['net'], optimize, rg[0], rg[-1], step), end='\n')   
    
    class_accuracy = []
    comp_accuracy = []
    
    if optimize == 'epoch':   
        model = create_Net(param)        
        epochs = get_range_param(rg, step)
        for i, e in enumerate(epochs):
            print("\r{} epochs {}".format(e, progress_bar(2*i, len(epochs)*2-1)), end='')
            # trains the model with the set param
            train_with_set_params(model, e, mini_batch_size)
            print("\r{} epochs {}".format(e, progress_bar(2*i+1, len(epochs)*2-1)), end='')
            # Tests the accuracy
            metrics = test_model(model, test_input, test_target, test_class)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])            
         
        save_figure(x_plot=epochs, y_plot=[class_accuracy, comp_accuracy], 
                legend=['Classification accuracy', 'Comparison accuracy'], 
                ax_labels=['# epochs', 'Accuracy'], title='', filename="GS{}{}.png".format(param['net'], optimize))
        
    elif optimize == 'hidden_layer_size':
        sizes = get_range_param(rg, step)
        for i, s in enumerate(sizes):
            print("\rhd size: {} {}".format(s, progress_bar(2*i, len(sizes)*2-1)), end='')
            # Creates modle with s hidden layers
            param['hidden'] = s
            model = create_Net(param)
            train_with_set_params(model, nb_epochs, mini_batch_size)
            print("\rhd size: {} {}".format(s, progress_bar(2*i+1, len(sizes)*2-1)), end='')
            metrics = test_model(model, test_input, test_target, test_class)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])            
            
        save_figure(x_plot=sizes, y_plot=[class_accuracy, comp_accuracy], 
                    legend=['Classification accuracy', 'Comparison accuracy'], 
                    ax_labels=['# hidden layers', 'Accuracy'], title='', filename="GS{}{}.png".format(param['net'], optimize))        
            
    elif optimize == 'mini_batch_size':
        model = create_Net(param)
        for i, s in enumerate(rg):
            print("\rmb size: {} {}".format(s, progress_bar(2*i, len(rg)*2-1)), end='')
            train_with_set_params(model, nb_epochs, s)
            print("\rmb size: {} {}".format(s, progress_bar(2*i+1, len(rg)*2-1)), end='')
            # training batch size is irrelevant for the test
            metrics = test_model(model, test_input, test_target, test_class)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])
            
        save_figure(x_plot=rg, y_plot=[class_accuracy, comp_accuracy], 
                    legend=['Classification accuracy', 'Comparison accuracy'], 
                    ax_labels=['Mini batch size', 'Accuracy'], title='', filename="GS{}{}.png".format(param['net'], optimize))
        
    elif optimize == 'drop_proba':
        probas = get_range_param(rg, step)
        for i, p in enumerate(probas):
            param['drop_proba'][level] = p
            model = create_Net(param)
            print("\rdrop proba level {}: {}".format(level, progress_bar(2*i, len(probas)*2-1)), end='')
            train_with_set_params(model, nb_epochs, mini_batch_size)
            print("\rdrop proba level {}: {}".format(level, progress_bar(2*i+1, len(probas)*2-1)), end='')
            # training batch size is irrelevant for the test
            metrics = test_model(model, test_input, test_target, test_class)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])
            
        save_figure(x_plot=probas, y_plot=[class_accuracy, comp_accuracy], 
                    legend=['Classification accuracy', 'Comparison accuracy'], 
                    ax_labels=['Drop proba @ level {}', 'Accuracy'], title='', filename="GS{}{}.png".format(param['net'], optimize))
        
    else:
        raise NotImplementedError
        

def test_param(param):
    """
    Trains the net and prints the error rate for the given parameters
    """
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
    for i in [0, 1]:
        train_model(model, Variable(train_input[:, i, :, :].view(-1, 1, 14, 14)), 
                           Variable(train_class[:, i].long()), param['batch_size'], param['epochs'])
    
    model.train(False)
    #TODO this is temp otherwise 2x computation
#    nb_test_errors, _, _, _ = test_model(model, test_input, test_target, test_class)
    nb_test_errors = 0               
    for i in [0, 1]:
        nb_test_errors += compute_nb_errors(model, Variable(test_input[:, i, :, :].view(-1, 1, 14, 14)), 
                                            Variable(test_class[:, i].long()), 20)    
    
    print('Classification test error: {:0.2f}% {:d}/{:d}'.format( nb_test_errors / test_input.size(0) /2*100, 
                                                                  nb_test_errors, 2*test_input.size(0)))

    pred_class = torch.stack([model(Variable(test_input[:, 0, :, :].view(-1, 1, 14, 14))).data.max(1)[1], 
                              model(Variable(test_input[:, 1, :, :].view(-1, 1, 14, 14))).data.max(1)[1]], dim=1)
    
    print("Comparison error: {:0.2f}% {:d}/{:d}".format( (compare_digits(pred_class) - test_target.int()).abs().sum().item() /test_target.size(0)*100, 
          (compare_digits(pred_class) - test_target.int()).abs().sum().item(), 
          test_input.size(0)))
    

def boosted_Net(net_dicts):
    """
    Trains multiple nets and boosts the results to have better perfs
    """  
    print("Creating boosted prediction with {} nets".format(len(net_dicts)))
    train_input, train_target, train_class, test_input, test_target, test_class = \
        prologue.generate_pair_sets(N_PAIRS)
    
    # Normalization of the data
    train_input /= train_input.max()
    test_input /= test_input.max()
    
    pred_tr = []; pred_te = [];
    for i, dic in enumerate(net_dicts):
        print("Net {}/{}".format(i+1, len(net_dicts)))
        for k in dic.keys():
            print(" {}: {} |".format(k, dic[k]), end='')
        print('')
        
        # setting a random seed
        if dic['seed'] != None:
            torch.manual_seed(dic['seed'])
            
        model = create_Net(dic)
        # Training each of the models
        model.train(True)
        for i in [0, 1]:                
                train_model(model, Variable(train_input[:, i, :, :].view(-1, 1, 14, 14)), 
                                   Variable(train_class[:, i].long()), dic['batch_size'], dic['epochs'])
        model.train(False)
        
        print(test_model(model, test_input, test_target, test_class))        
        
        output_tr = []; output_te= [];
        for i in [0, 1]:
            out = model(Variable(train_input[:, i, :, :].view(-1, 1, 14, 14)))
            output_tr.append(out.data.max(1)[1])
            
            out = model(Variable(test_input[:, i, :, :].view(-1, 1, 14, 14)))
            output_te.append(out.data.max(1)[1])
            
        pred_tr.append(torch.cat(output_tr))
        pred_te.append(torch.cat(output_te))

    target = torch.cat([train_class[:,0], train_class[:,1]])
    boost_model = xgb.XGBClassifier().fit(torch.stack(pred_tr, 1), target)
    boost_pred = boost_model.predict(torch.stack(pred_te, 1))
    
    pred_classes = torch.Tensor(np.stack([boost_pred[:1000], boost_pred[1000:]], axis=1))
    print("Classification error w/ boosting: {:.4}%".format((boost_pred != torch.cat([test_class[:,0], test_class[:,1]]).numpy()).sum() /boost_pred.shape[0]*100))
    print("Comparison error w/ boosting: {:.4}%\n".format((compare_digits(pred_classes) - test_target.int()).abs().sum().item() / test_target.size(0)*100))


















































