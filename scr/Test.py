#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:39:02 2020

@author: andyq
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report


def Test(model, test_data, test_label, GPU_device):
    test_output = model(test_data)
    if GPU_device == True:
        pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()
    else:
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    Acc = np.mean(pred_y == test_label)
    Conf_Mat = confusion_matrix(test_label, pred_y)  # confusion matrix
    Acc_N = Conf_Mat[0][0]/np.sum(Conf_Mat[0])
    Acc_A = Conf_Mat[1][1]/np.sum(Conf_Mat[1])
    Acc_V = Conf_Mat[2][2]/np.sum(Conf_Mat[2])
    Acc_R = Conf_Mat[3][3]/np.sum(Conf_Mat[3])
    Acc_P = Conf_Mat[4][4]/np.sum(Conf_Mat[4])
    Acc_L = Conf_Mat[5][5]/np.sum(Conf_Mat[5])
    
    print('\nAccuracy=%.2f%%'%(Acc*100))
    print('Accuracy_N=%.2f%%'%(Acc_N*100))
    print('Accuracy_A=%.2f%%'%(Acc_A*100))
    print('Accuracy_V=%.2f%%'%(Acc_V*100))
    print('Accuracy_R=%.2f%%'%(Acc_R*100))
    print('Accuracy_P=%.2f%%'%(Acc_P*100))
    print('Accuracy_L=%.2f%%'%(Acc_L*100))
    print('\nConfusion Matrix:\n')
    print(Conf_Mat)
    
    print("======================================")
    
    report = classification_report(test_label,pred_y)
    print(report)


def TestGRUAttention(encoder,decoder, test_data, test_label, GPU_device):
    encoder_outputs, encoder_hidden = encoder(test_data)
    decoder_hidden = encoder_hidden
    test_output, decoder_hidden = decoder(test_data, decoder_hidden, encoder_outputs)
    if GPU_device == True:
         pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()
    else:
         pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    Acc = np.mean(pred_y == test_label)
    Conf_Mat = confusion_matrix(test_label, pred_y)  # confusion matrix
    
    Acc_N = Conf_Mat[0][0] / np.sum(Conf_Mat[0])
    Acc_A = Conf_Mat[1][1] / np.sum(Conf_Mat[1])
    Acc_V = Conf_Mat[2][2] / np.sum(Conf_Mat[2])
    Acc_R = Conf_Mat[3][3] / np.sum(Conf_Mat[3])
    Acc_P = Conf_Mat[4][4] / np.sum(Conf_Mat[4])
    Acc_L = Conf_Mat[5][5] / np.sum(Conf_Mat[5])
    print('\n Accuracy:\n')
    print('\nAccuracy=%.2f%%' % (Acc * 100))
    print('Accuracy_N=%.2f%%' % (Acc_N * 100))
    print('Accuracy_A=%.2f%%' % (Acc_A * 100))
    print('Accuracy_V=%.2f%%' % (Acc_V * 100))
    print('Accuracy_R=%.2f%%' % (Acc_R * 100))
    print('Accuracy_P=%.2f%%' % (Acc_P * 100))
    print('Accuracy_L=%.2f%%' % (Acc_L * 100))
    print("======================================")
    print('\nConfusion Matrix:\n')
    print(Conf_Mat)
    
    print("======================================")
    
    report = classification_report(test_label,pred_y)
    print(report)