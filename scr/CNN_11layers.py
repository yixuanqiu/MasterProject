#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 22:03:00 2020

@author: andyq
"""

import torch.nn as nn

class CNN_11(nn.Module):
    def __init__(self):
        super(CNN_11,self).__init__()
        self.conv1 = nn.Conv1d(1,3,kernel_size=27,stride=1)
        self.conv2 = nn.Conv1d(3,10,kernel_size=15, stride=1)
        self.conv3 = nn.Conv1d(10,10,kernel_size=4,stride=1)
        self.conv4 = nn.Conv1d(10,10,kernel_size=3,stride=1)

        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(730,30)
        self.fc2 = nn.Linear(30,10)
        self.fc3 = nn.Linear(10,6)


    def forward(self,x):
        batch_size = x.size(0)
        x = self.pooling(self.conv1(x))
        x = self.pooling(self.conv2(x))
        x = self.pooling(self.conv3(x))
        x = self.pooling(self.conv4(x))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x