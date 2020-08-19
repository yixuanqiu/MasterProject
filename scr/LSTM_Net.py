#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:39:49 2020

@author: andyq
"""

import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(LSTM,self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, 4, batch_first=True)

        self.out = nn.Linear(hidden_size, 6)


    def forward(self, x):
        r_out,(h_n,h_c) = self.rnn(x, None)
        out = self.out(r_out[:,-1,:])
        return out
