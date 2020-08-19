#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:45:09 2020

@author: andyq
"""
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv1d(1,8,kernel_size=5)
        self.conv2 = nn.Conv1d(8,16,kernel_size=5)
        self.conv3 = nn.Conv1d(16,32,kernel_size=5)
        self.pooling = nn.MaxPool1d(2)
        self.fc = nn.Linear(1088,6)

    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = x.view(batch_size, -1)
        x= self.fc(x)
        return x