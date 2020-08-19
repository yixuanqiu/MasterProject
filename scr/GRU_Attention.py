#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:01:36 2020

@author: andyq
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, self.n_layers, batch_first=True, bidirectional=True)

    def forward(self, input):
        output, hidden = self.gru(input)
        output = output.transpose(0, 1)
        hidden = torch.cat((hidden[0], hidden[1]), 1)
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2, dropout_p=0.3, max_length=1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU((hidden_size*2)+input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,input, hidden, encoder_outputs):
        batch_size = input.size(0)
        attn_weights = F.softmax(self.attn(hidden), dim=1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_weights = attn_weights.unsqueeze(1)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((input, attn_applied),dim=2)
        output,hidden = self.gru(rnn_input)
        # attn_applied = attn_applied.squeeze(1)
        # # output = torch.cat((input[0], attn_applied[0]), 1)
        # output = self.attn_combine(attn_applied)
        output = torch.sigmoid(output)
        output = self.out(output)
        output = output.squeeze(0)
        output = output.view(batch_size, -1)
        return output, hidden