#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:39:02 2020

@author: andyq
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score


def Test(model, test_data, test_label, GPU_device):
    test_output = model(test_data)
    if GPU_device:
        pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()
    else:
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    get_results(pred_y, test_label)

def evaluation(model, validate_loader):
    target_pred = []
    target_true = []
    with torch.no_grad():
        for data in validate_loader:
            inputs, target = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            target_pred += predicted.data.tolist()
            target_true += target.data.tolist()
    get_results(target_pred, target_true)


def get_results(target_pred, target_true):
    Acc = accuracy_score(target_true, target_pred)
    Conf_Mat = confusion_matrix(target_true, target_pred)  # confusion matrix
    Acc_N = Conf_Mat[0][0] / np.sum(Conf_Mat[0])
    Acc_A = Conf_Mat[1][1] / np.sum(Conf_Mat[1])
    Acc_V = Conf_Mat[2][2] / np.sum(Conf_Mat[2])
    Acc_R = Conf_Mat[3][3] / np.sum(Conf_Mat[3])
    Acc_P = Conf_Mat[4][4] / np.sum(Conf_Mat[4])
    Acc_L = Conf_Mat[5][5] / np.sum(Conf_Mat[5])

    print('\nAccuracy=%.2f%%' % (Acc * 100))
    print('Accuracy_N=%.2f%%' % (Acc_N * 100))
    print('Accuracy_A=%.2f%%' % (Acc_A * 100))
    print('Accuracy_V=%.2f%%' % (Acc_V * 100))
    print('Accuracy_R=%.2f%%' % (Acc_R * 100))
    print('Accuracy_P=%.2f%%' % (Acc_P * 100))
    print('Accuracy_L=%.2f%%' % (Acc_L * 100))
    print('\nConfusion Matrix:\n')
    print(Conf_Mat)

    print("======================================")
    report = classification_report(target_true, target_pred)
    print(report)


def gru_evaluation(encoder, decoder, validate_loader):
    target_pred = []
    target_true = []
    with torch.no_grad():
        for data in validate_loader:
            inputs, target = data
            encoder_outputs, encoder_hidden = encoder(inputs)
            decoder_hidden = encoder_hidden
            decoder_output, decoder_hidden = decoder(inputs, decoder_hidden, encoder_outputs)

            _, predicted = torch.max(decoder_output, dim=1)
            target_pred += predicted.data.tolist()
            target_true += target.data.tolist()
    get_results(target_pred, target_true)
