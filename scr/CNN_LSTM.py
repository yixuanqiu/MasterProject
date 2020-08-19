#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 22:08:25 2020

@author: andyq
"""
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM,self).__init__()
        self.conv1 = nn.Conv1d(1,3,kernel_size=20,stride=1,padding=19)
        self.conv2 = nn.Conv1d(3,6,kernel_size=10, stride=1,padding=9)
        self.conv3 = nn.Conv1d(6,6,kernel_size=5,stride=1, padding=4)
        self.lstm = nn.LSTM(131,20,2,dropout = 0.3, batch_first = True)


        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)


        self.fc1 = nn.Linear(20,20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,6)


    def forward(self,x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x, (h_n, h_c) = self.lstm(x,None)
        x = self.dropout(self.fc1(x[:, -1, :]))
        x = self.dropout(self.fc2(x))
        x = self.fc3(x)
        return x