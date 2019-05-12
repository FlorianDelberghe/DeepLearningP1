# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import optim
from torch import nn
import os

import torchvision
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
from Utilities import progress_bar
from NetUtilities import *
from CompNets import *

def main():
    
    torch.manual_seed(999)
    
    # Best params for the nets
    cnet2 = {"net": 'compNet2', "hidden": 120, "epochs": 80, "batch_size": 10, 
             "pool": 'max', "activation": 'relu', "naive": False, "norm": 'dropout', 
             "drop_proba": [0.05, 0.05, 0.5, 0.2, 0, 0, 0], "seed": None}
    cnet4 = {"net": 'compNet4', "hidden": 350, "epochs": 30, "batch_size": 10, 
             "pool": 'max', "activation": 'tanh', "naive": False, "norm": 'dropout', 
             "drop_proba": [0, 0, 0, 0.2, 0, 0, 0], "seed": None}   
    cnet5 =  {"net": 'compNet5', "hidden": 120, "epochs": 30, "batch_size": 20, 
             "pool": 'max', "activation": 'relu', "naive": False, "norm": 'dropout', 
             "drop_proba": [0.0, 0.0, 0.0, 0.0, 0.0], "seed": None}
    
    test_param(cnet2, save=False, log=False)
    test_param(cnet4, save=False, log=False)
    test_param(cnet5, save=False, log=False)
    
    for _ in range(10):
        # Testing the performances of the nets 10 times to compute variance
        test_param(cnet4, save=True, log=False)

if __name__ == '__main__':
    main()


















































































