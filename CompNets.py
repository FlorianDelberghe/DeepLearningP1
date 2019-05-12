# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

class compNet2(nn.Module):
    """
    LeNet model
    """
    def __init__(self, param):
        super(compNet2, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) #32*10*10
        self.drop_conv1 = nn.Dropout2d(param['drop_proba'][0])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) #64*4*4
        self.drop_conv2 = nn.Dropout2d(param['drop_proba'][1])
        self.fc1 = nn.Linear(256, param['hidden']) #64*2*2=256 -> 120
        self.drop1 = nn.Dropout(param['drop_proba'][2])
        self.fc2 = nn.Linear(param['hidden'], 10) # 120 -> 10
        self.drop2 = nn.Dropout(param['drop_proba'][3])
        
        self.naive = param['naive']
        if self.naive:
            # naive comp net
            self.fcnaive = nn.Linear(2, 2)        
        else:   
            # fc net for comp
            self.drop3 = nn.Dropout(param['drop_proba'][4])
            self.fc3 = nn.Linear(20, 60)
            self.drop4 = nn.Dropout(param['drop_proba'][5])
            self.fc4 = nn.Linear(60, 90)
            self.drop5 = nn.Dropout(param['drop_proba'][6])
            self.fc5 = nn.Linear(90,2)        
        
        if param['activation'] == 'tanh':
            self.activation = torch.tanh
        elif param['activation'] == 'relu':
            self.activation = F.relu
        else :
            raise ValueError("Activation must be 'relu' or 'tanh'")
            
        if param['pool'] == 'max':
            self.pool = F.max_pool2d
        elif param['pool'] == 'avg':
            self.pool = F.avg_pool2d
        else :
            raise ValueError("Activation must be 'avg' or 'max'")
        
    def forward(self, input_):        
        
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        x = self.activation(self.pool(self.drop_conv1(self.conv1(x)), kernel_size=2, stride=2))
        x = self.activation(self.pool(self.drop_conv2(self.conv2(x)), kernel_size=2, stride=2))
        x = self.activation(self.fc1(self.drop1(x.view(-1, 256))))
        x = self.fc2(self.drop2(x))
        
        y = self.activation(self.pool(self.drop_conv1(self.conv1(y)), kernel_size=2, stride=2))
        y = self.activation(self.pool(self.drop_conv2(self.conv2(y)), kernel_size=2, stride=2))
        y = self.activation(self.fc1(self.drop1(y.view(-1, 256))))
        y = self.fc2(self.drop2(y))

        if self.naive:
            # Equivalent to the noive comparing digits method
            z = torch.stack([x.data.max(1)[1].float(), y.data.max(1)[1].float()], dim=1)
            z = self.fcnaive(z)
        else:
            z = torch.cat([x, y], 1)          
            z = F.relu(self.fc3(self.drop3(z)))
            z = F.relu(self.fc4(self.drop4(z)))
            z = self.fc5(self.drop5(z))       
            
        return x, y, z
    
    
class compNet4(nn.Module):
    """
    LeNEt4 based model
    """
    def __init__(self, param):
        super(compNet4, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3) #4*12*12
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3) #16*10*10
        self.conv3 = nn.Conv2d(16, 120, kernel_size=2) #120*1*1
        self.fc1 =  nn.Linear(120, param['hidden']) #flat fc 120->350
        self.drop1 = nn.Dropout(param['drop_proba'][2])
        self.fc2 = nn.Linear(param['hidden'], 10) #flat fc 350->10
        self.drop2 = nn.Dropout(param['drop_proba'][3])
        
        self.naive = param['naive']
        if self.naive:
            # naive comp net
            self.fcnaive = nn.Linear(2, 2)        
        else:   
            # fc net for comp
            self.drop3 = nn.Dropout(param['drop_proba'][4])
            self.fc3 = nn.Linear(20, 60)
            self.drop4 = nn.Dropout(param['drop_proba'][5])
            self.fc4 = nn.Linear(60, 90)
            self.drop5 = nn.Dropout(param['drop_proba'][6])
            self.fc5 = nn.Linear(90,2)        
        
        if param['activation'] == 'tanh':
            self.activation = torch.tanh
        elif param['activation'] == 'relu':
            self.activation = F.relu
        else :
            raise ValueError("Activation must be 'relu' or 'tanh'")
            
        if param['pool'] == 'max':
            self.pool = F.max_pool2d
        elif param['pool'] == 'avg':
            self.pool = F.avg_pool2d
        else :
            raise ValueError("Pooling must be 'avg' or 'max'")
        
    def forward(self, input_):        
        
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        x = self.activation(self.pool(self.conv1(x), kernel_size=2, stride=2)) 
        x = self.activation(self.pool(self.conv2(x), kernel_size=2, stride=2))
        x = self.activation(self.conv3(x))
        x = self.activation(self.fc1(x.view(-1, 120)))
        x = self.fc2(x)
        
        y = self.activation(self.pool(self.conv1(y), kernel_size=2, stride=2)) 
        y = self.activation(self.pool(self.conv2(y), kernel_size=2, stride=2))
        y = self.activation(self.conv3(y))
        y = self.activation(self.fc1(y.view(-1, 120)))
        y = self.fc2(y)
        
        if self.naive:
            # Equivalent to the noive comparing digits method
            z = torch.stack([x.data.max(1)[1].float(), y.data.max(1)[1].float()], dim=1)
            z = self.fcnaive(z)
        else:
            z = torch.cat([x, y], 1)            
            z = F.relu(self.fc3(self.drop3(z)))
            z = F.relu(self.fc4(self.drop4(z)))
            z = self.fc5(self.drop5(z))   
        
        return x, y, z
    
    
class compNet5(nn.Module):
    """
    LeNEt5 model
    """
    def __init__(self, param):
        super(compNet5, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=2)
        self.fc1 =  nn.Linear(120, 120)
        self.drop1 = nn.Dropout(param['drop_proba'][2])
        self.fc2 = nn.Linear(120, param['hidden'])
        self.drop2 = nn.Dropout(param['drop_proba'][3])
        self.fc3 = nn.Linear(param['hidden'], 10)
        self.drop3 = nn.Dropout(param['drop_proba'][4])
        
        self.naive = param['naive']
        if self.naive:
            # naive comp net
            self.fcnaive = nn.Linear(2, 2)        
        else:   
            # fc net for comp
            self.drop4 = nn.Dropout(0.0)
            self.fc4 = nn.Linear(20, 60)
            self.drop5 = nn.Dropout(0.0)
            self.fc5 = nn.Linear(60, 90)
            self.drop6 = nn.Dropout(0.0)
            self.fc6 = nn.Linear(90,2)        
            
        
        if param['activation'] == 'tanh':
            self.activation = torch.tanh
        elif param['activation'] == 'relu':
            self.activation = F.relu
        else :
            raise ValueError("Activation must be 'relu' or 'tanh'")
            
        if param['pool'] == 'max':
            self.pool = F.max_pool2d
        elif param['pool'] == 'avg':
            self.pool = F.avg_pool2d
        else :
            raise ValueError("Activation must be 'avg' or 'max'")
    
        
    def forward(self, input_):        
        
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        x = self.activation(self.pool(self.conv1(x), kernel_size=2, stride=2))
        x = self.activation(self.pool(self.conv2(x), kernel_size=2, stride=2))
        x = self.activation(self.conv3(x))
        x = self.activation(self.fc1(self.drop1(x.view(-1, 120))))
        x = self.activation(self.fc2(self.drop2(x)))        
        x = self.fc3(self.drop3(x))
        
        y = self.activation(self.pool(self.conv1(y), kernel_size=2, stride=2))
        y = self.activation(self.pool(self.conv2(y), kernel_size=2, stride=2))
        y = self.activation(self.conv3(y))
        y = self.activation(self.fc1(self.drop1(y.view(-1, 120))))
        y = self.activation(self.fc2(self.drop2(y)))
        y = self.fc3(self.drop3(y))
        
        if self.naive:
            # Equivalent to the noive comparing digits method
            z = torch.stack([x.data.max(1)[1].float(), y.data.max(1)[1].float()], dim=1)
            z = self.fcnaive(z)
        else:
            z = torch.cat([x, y], 1)         
            z = F.relu(self.fc4(self.drop4(z)))
            z = F.relu(self.fc5(self.drop5(z)))
            z = self.fc6(self.drop6(z))       
        
        return x, y, z


class resNet(nn.Module):
    
    def __init__():
        raise NotImplementedError
        
    def forward():
        raise NotImplementedError


def create_Net(param):
    """
    Return a Net from one of the availables classes
    """        
    if param['net'] == 'Net1':
        return Net1(param)
    elif param['net'] == 'Net2': 
        return Net2(param)
    elif param['net'] == 'LeNet4':
        return LeNet4(param)
    elif param['net'] == 'LeNet5':
        return LeNet5(param)
    elif param['net'] == 'compNet2':
        return compNet2(param)
    elif param['net'] == 'compNet4':
        return compNet4(param)
    elif param['net'] == 'compNet5':
        return compNet5(param)
    else:
        raise NotImplementedError






































































