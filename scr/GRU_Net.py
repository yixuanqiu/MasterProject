#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:17:04 2020

@author: andyq
"""

import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, max_length = 5):
        super(GRU, self).__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.output_size = output_size

        self.GRU = nn.GRU(
            input_size,
            hidden_size,
            self.n_layers,
            batch_first=True,
            dropout=0.5

        )
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        output, hidden = self.GRU(input)
        out = self.out(output)
        output = out[:,-1,:]
        return output
