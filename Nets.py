# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

class Net1(nn.Module):
    """
    First NN model based on Practice 4
    """
    def __init__(self, nb_hidden):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
    
class Net2(nn.Module):
    """
    NN model similar to Net1 w/ dropout
    """
    def __init__(self, nb_hidden):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.drop_conv1 = nn.Dropout2d(0.05)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.drop_conv2 = nn.Dropout2d(0.05)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.drop2 = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.drop_conv1(self.conv1(x)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.drop_conv2(self.conv2(x)), kernel_size=2, stride=2))
        x = F.relu(self.fc1(self.drop1(x.view(-1, 256))))
        x = self.fc2(self.drop2(x))
        return x


class LeNet4(nn.Module):
    """
    LeNEt4 model
    """
    def __init__(self, nb_hidden=120):
        super(LeNet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3) #4*12*12
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3) #16*10*10
        self.conv3 = nn.Conv2d(16, 120, kernel_size=2) #120*1*1
        self.fc1 =  nn.Linear(120, nb_hidden) #flat fc 120->120
        self.fc2 = nn.Linear(nb_hidden, 10) #flat fc 120->10
        
    def forward(self, x):
        x = F.tanh(F.avg_pool2d(self.conv1(x), kernel_size=2, stride=2)) 
        x = F.tanh(F.avg_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.fc1(x.view(-1, 120)))
        x = self.fc2(x)
        return x
    
    
class LeNet5(nn.Module):
    """
    LeNEt5 model
    """
    def __init__(self, nb_hidden=84):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=2)
        self.fc1 =  nn.Linear(120, 120)
        self.fc2 = nn.Linear(120, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 10)
        
    def forward(self, x):
        x = F.tanh(F.avg_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.tanh(F.avg_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.fc1(x.view(-1, 120)))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
