#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 22:07:06 2020

@author: andyq
"""

import torch.nn as nn
import torch.nn.functional as F


class CNN_16(nn.Module):
    def __init__(self):
        super(CNN_16,self).__init__()
        self.conv1 = nn.Conv1d(1,128,kernel_size=25,stride=3)
        self.conv2 = nn.Conv1d(128,32,kernel_size=3,stride=1)
        self.conv3 = nn.Conv1d(32,32,kernel_size=5,stride=1)
        self.conv4 = nn.Conv1d(32,128,kernel_size=2,stride=2)
        self.conv5 = nn.Conv1d(128,256,kernel_size=7,stride=1)
        self.conv6 = nn.Conv1d(256,512,kernel_size=3,stride=1)
        self.conv7 = nn.Conv1d(512,128,kernel_size=3,stride=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512,30)
        self.fc2 = nn.Linear(30,6)

        self.dropout = nn.Dropout(0.1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self,x):
        batch_size = x.size(0)
        x = self.maxpool1(self.bn1(F.relu(self.conv1(x))))
        x = self.maxpool2(self.bn2(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(F.relu(self.conv4(x)))
        x = self.maxpool2(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x