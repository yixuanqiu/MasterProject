#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:29:19 2020

@author: andyq
"""
import numpy as np
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import scale
from scipy.signal import resample


def LoadData(path):
        data_file = '/Data.npy'
        label_file = '/Label.npy'
        data = np.load(path+data_file)
        data.astype(np.float32)
        label = np.load(path+label_file)
        return data, label

def LoadSegData(path):
    data_all = np.load(path)
    data_all.astype(np.float32)
    label = data_all[:,1800]
    data = data_all[:,0:1800]
    return data, label
    
def ShuffleData(data, label,sep):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        
        # divide training and testing dataset
        sep = int(sep*len(data))
        train_data = data[indices[:sep]]
        train_label = label[indices[:sep]]
        test_data = data[indices[sep:]]
        test_label = label[indices[sep:]]
        
        train_label = np.array(train_label)
        train_label.astype(np.long)
        
        return train_data, train_label, test_data, test_label
        
def Nomalization(train_data):
        min_max_scaler = preprocessing.MinMaxScaler()
        train_data = min_max_scaler.fit_transform(train_data)
        return train_data
    
def AsTorch(test_data):
    test_data = torch.from_numpy(test_data)
    test_data = torch.unsqueeze(test_data, dim=1).type(torch.FloatTensor)
    return test_data


# Construct Train Dataset
class TrainDataset(Data.Dataset):
    def __init__(self, train_data, train_label):
        data = train_data
        label = train_label
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data)
        self.x_data = torch.unsqueeze(self.x_data, dim=1).type(torch.FloatTensor)
        self.y_data =torch.from_numpy(label).type(torch.long)
    
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

# Construct Test Dataset
class TestDataset(Data.Dataset):
    def __init__(self, test_data, test_label):
        self.x_data = test_data
        label = test_label
        self.len = test_data.shape[0]
        self.y_data = torch.from_numpy(label).type(torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
def buildCNN(Network, LR, device):
    model = Network()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR)
    criterion = nn.CrossEntropyLoss()
        
    Device = device
    if Device ==True:
        model.cuda()
    return model, optimizer, criterion

def buildLSTM(Network,input_size, hidden_size, lr, device):
    model = Network(input_size, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(),lr)
    criterion = nn.CrossEntropyLoss()
    Device = device
    if Device ==True:
        model.cuda()
    return model, optimizer, criterion

def buildGRU(Network,input_size, hidden_size, output_size, lr, device):
    model = Network(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(),lr)
    criterion = nn.CrossEntropyLoss()
    Device = device
    if Device ==True:
        model.cuda()
    return model, optimizer, criterion

def buildGRU_Attention(Encoder,AttnDecoderRNN,input_size, hidden_size, lr, device):
    encoder = Encoder(input_size, hidden_size)
    decoder = AttnDecoderRNN(input_size, hidden_size, 6, 1)
    Device = device
    if Device ==True:
        encoder.cuda()
        decoder.cuda()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr)
    criterion = nn.CrossEntropyLoss()  
    return encoder, decoder, encoder_optimizer, decoder_optimizer, criterion   
        
def TrainLoader(train_data, train_label, batch_size, shuffle, num_workers):
    trainset = TrainDataset(train_data, train_label)
    train_loader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader

def TestLoader(test_data, test_label, batch_size, shuffle):
    testset = TestDataset(test_data, test_label)
    test_loader = Data.DataLoader(testset, batch_size = batch_size, shuffle=shuffle)
    return test_loader
    
def ConstructSegData(path, batch_size,sep):
    data, label = LoadSegData(path)
    train_data, train_label, test_data, test_label = ShuffleData(data,label,sep)
    # train_data = scale(train_data, axis = 1)
    # test_data =scale(test_data, axis =1)
    test_data = AsTorch(test_data)
    train_loader = TrainLoader(train_data, train_label, batch_size, True, 3)
    test_loader = TestLoader(test_data, test_label, batch_size, True)
    return train_loader, test_loader, test_data, test_label
    
def ConstructHeartBeatData(path, batch_size,sep):
    data, label = LoadData(path)
    train_data, train_label, test_data, test_label = ShuffleData(data,label,sep)
    test_data = AsTorch(test_data)
    train_loader = TrainLoader(train_data, train_label, batch_size, True, 3)
    test_loader = TestLoader(test_data, test_label, batch_size, True)
    return train_loader, test_loader, test_data, test_label

def ConstructResample(path, resample_size, batch_size,sep):
    data, label = LoadSegData(path)
    data = resample(data,resample_size,axis=1)
    data = scale(data, axis=1)
    train_data, train_label, test_data, test_label = ShuffleData(data,label,sep)
    train_data = scale(train_data, axis = 1)
    test_data = AsTorch(test_data)
    train_loader = TrainLoader(train_data, train_label, batch_size, True, 3)
    test_loader = TestLoader(test_data, test_label, batch_size, True)
    return train_loader, test_loader, test_data, test_label

def GetMethodIndex():
    num = int(input("please choose the number: \n \
        - PreProcessing：\n \
          0. Preprocessing for hearbeat dataset \n \
        - Based on Heart Beats: \n \
          1. SVM \n \
          2. Xgboost \n \
          3. WT-CNN \n \
          4. WT-LSTM \n \
        - Based on Segments: \n \
          5. LSTM \n \
          6. GRU \n \
          7. GRU with Attention \n \
        - Reproduce \n \
          8. 11 layers CNN Network \n \
          9. 16 Lyaers CNN Network \n \
          10 .CNN-LSTM Network \n \
              "))
    return num