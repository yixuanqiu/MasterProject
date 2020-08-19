#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 20:28:47 2020

@author: andyq
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import sklearn
import matplotlib.pyplot as plt

class SVM():
    
    @staticmethod
    def LoadData(path):
        Feature = np.load(path + '/feature_Data.npy')
        Label = np.load(path + '/feature_Label.npy')
        Label = Label.squeeze()
        
        return Feature, Label
    

    @staticmethod
    def Noramalization(Feature, Label):
        train_x, test_x, train_y, test_y = train_test_split(Feature, Label, test_size=0.7, random_state=42)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_x = min_max_scaler.fit_transform(train_x)
        test_x = min_max_scaler.transform(test_x)
        return train_x, test_x, train_y, test_y
    
        
    @staticmethod
    def train_SVM(train_x, train_y):
        print('SVM training...')
        SVM_model = SVC(kernel='rbf', C=4, gamma=4)
        SVM_model.fit(train_x, train_y)
        return SVM_model
    
    
    @staticmethod    
    def Prediction(SVM_model, test_x, test_y):
        y_predict = SVM_model.predict(test_x)
        return y_predict
    
    
    @staticmethod
    def Test(y_predict, test_y):
        print('SVM testing...')
        Acc = np.mean(y_predict == test_y)
        Conf_Mat = confusion_matrix(test_y, y_predict)  # confusion matrix
        f1 = f1_score(test_y, y_predict, average=  'micro')
        Acc_N = Conf_Mat[0][0]/np.sum(Conf_Mat[0])
        Acc_A = Conf_Mat[1][1]/np.sum(Conf_Mat[1])
        Acc_V = Conf_Mat[2][2]/np.sum(Conf_Mat[2])
        Acc_R = Conf_Mat[3][3]/np.sum(Conf_Mat[3])
        Acc_P = Conf_Mat[4][4]/np.sum(Conf_Mat[4])
        Acc_L = Conf_Mat[5][5]/np.sum(Conf_Mat[5])
        
        print('\nAccuracy=%.2f%%'%(Acc*100))
        print('F1 Score = %.2f%%'%(f1*100))
        print('Accuracy_N=%.2f%%'%(Acc_N*100))
        print('Accuracy_A=%.2f%%'%(Acc_A*100))
        print('Accuracy_V=%.2f%%'%(Acc_V*100))
        print('Accuracy_R=%.2f%%'%(Acc_R*100))
        print('Accuracy_P=%.2f%%'%(Acc_P*100))
        print('Accuracy_L=%.2f%%'%(Acc_L*100))
        print('\nConfusion Matrix:\n')
        print(Conf_Mat)
    
    @staticmethod    
    def GetReport(test_y, y_predict):
        report = sklearn.metrics.classification_report(test_y,y_predict)
        print(report)
        
    @staticmethod
    def PLotConfusionMatrix(SVM_model, test_x, test_y):
        sklearn.metrics.plot_confusion_matrix(SVM_model,test_x,test_y)
        plt.show()
        
        
# path = '/Users/andyq/Documents/Code/ECGproject/features/'
# Feature, Label = SVM.LoadData(path)
# train_x, test_x, train_y, test_y = SVM.Noramalization(Feature,Label)
# SVM_model = SVM.train_SVM(train_x, train_y)
# y_predict = SVM.Prediction(SVM_model,test_x, test_y)
# SVM.Test(SVM_model, y_predict, test_y)
# SVM.GetReport(test_y, y_predict)


        