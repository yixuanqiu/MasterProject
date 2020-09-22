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
import os


def LoadData(path):
    data_file = '/Data.npy'
    label_file = '/Label.npy'
    data = np.load(path + data_file)
    data.astype(np.float32)
    label = np.load(path + label_file)
    return data, label


def LoadSegData(train_path, test_path, val_path):
    train_all = np.load(train_path)
    test_all = np.load(test_path)
    validate_all = np.load(val_path)

    return train_all, test_all, validate_all


def get_data(data):
    load_data = data[:, 0:1800]

    return load_data


def get_label(data):
    load_label = data[:, 1800]

    return load_label


def TrainData(train_all):
    train_data = get_data(train_all)
    train_label = get_label(train_all)
    train_label = np.array(train_label)
    train_label.astype(np.long)

    return train_data, train_label


def TestData(test_all):
    test_data = get_data(test_all)
    test_label = get_label(test_all)

    return test_data, test_label


def ValidateData(validate_all):
    validate_data = get_data(validate_all)
    validate_label = get_label(validate_all)

    return validate_data, validate_label


def ShuffleData(data, label, sep):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    # divide training and testing dataset
    sep = int(sep * len(data))
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


def AsTorch(test_data, gpu=True):
    test_data = torch.from_numpy(test_data)
    if (gpu):
        test_data = torch.unsqueeze(test_data, dim=1).type(torch.FloatTensor).cuda()
    else:
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
        self.y_data = torch.from_numpy(label).type(torch.long)

    def __getitem__(self, index):
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


# Construct Validate Dataset
class ValidateDataset(Data.Dataset):
    def __init__(self, validate_data, validate_label):
        self.x_data = validate_data
        label = validate_label
        self.len = validate_data.shape[0]
        self.y_data = torch.from_numpy(label).type(torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def buildCNN(Network, LR, device):
    model = Network()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    Device = device
    if Device:
        model.cuda()
    return model, optimizer, criterion


def buildLSTM(Network, input_size, hidden_size, lr, device):
    model = Network(input_size, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    Device = device
    if Device:
        model.cuda()
    return model, optimizer, criterion


def buildGRU(Network, input_size, hidden_size, output_size, lr, device):
    model = Network(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    Device = device
    if Device:
        model.cuda()
    return model, optimizer, criterion


def buildGRU_Attention(Encoder, AttnDecoderRNN, input_size, hidden_size, lr, device):
    encoder = Encoder(input_size, hidden_size)
    decoder = AttnDecoderRNN(input_size, hidden_size, 6, 1)
    Device = device
    if Device:
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
    test_loader = Data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    return test_loader


def ValidateLoader(validate_data, validate_label, batch_size, shuffle):
    valideteset = ValidateDataset(validate_data, validate_label)
    validate_loader = Data.DataLoader(valideteset, batch_size=batch_size, shuffle=shuffle)

    return validate_loader


def ConstructSegData(train_path, test_path, validate_path, batch_size, num_workers, z_scale, gpu):
    """
    train_path: the path of training set
    test_path: the path of testing set
    validate_path: the path of validate set
    batch_size:
    number_workers: parallel working
    scale: z-transform
    gpu:
    """
    train_all, test_all, validate_all = LoadSegData(train_path, test_path, validate_path)
    train_data, train_label = TrainData(train_all)
    test_data, test_label = TestData(test_all)
    validate_data, validate_label = ValidateData(validate_all)
    # train_data, train_label, test_data, test_label = ShuffleData(data, label, sep)
    if z_scale:
        train_data = scale(train_data, axis=1)
        test_data = scale(test_data, axis=1)
        validate_data = scale(validate_data, axis=1)
    test_data = AsTorch(test_data, gpu)
    validate_data = AsTorch(validate_data, gpu)
    train_loader = TrainLoader(train_data, train_label, batch_size, True, num_workers=num_workers)
    test_loader = TestLoader(test_data, test_label, batch_size, True)
    validate_loader = ValidateLoader(validate_data, validate_label, batch_size, True)

    return train_loader, test_loader, validate_loader, test_data, test_label


def ConstructHeartBeatData(path, batch_size, sep, num_workers):
    data, label = LoadData(path)
    train_data, train_label, test_data, test_label = ShuffleData(data, label, sep)
    test_data = AsTorch(test_data, gpu=False)
    train_loader = TrainLoader(train_data, train_label, batch_size, True, num_workers=num_workers)
    test_loader = TestLoader(test_data, test_label, batch_size, True)

    return train_loader, test_loader, test_data, test_label


def ConstructResample(train_path, test_path, validate_path, resample_size, batch_size, num_workers, gpu):
    train_all, test_all, validate_all = LoadSegData(train_path, test_path, validate_path)
    train_data, train_label = TrainData(train_all)
    test_data, test_label = TestData(test_all)
    validate_data, validate_label = ValidateData(validate_all)
    train_data = resample(train_data, resample_size, axis=1)
    train_data = scale(train_data, axis=1)
    test_data = resample(test_data, resample_size, axis=1)
    test_data = scale(test_data, axis=1)
    validate_data = resample(train_data, resample_size, axis=1)
    validate_data = scale(train_data, axis=1)
    # train_data, train_label, test_data, test_label = ShuffleData(data, label, sep)
    train_data = scale(train_data, axis=1)
    test_data = scale(test_data)
    validate_data = scale(validate_data)
    test_data = AsTorch(test_data, gpu)
    validate_data = AsTorch(validate_data, gpu)
    train_loader = TrainLoader(train_data, train_label, batch_size, True, num_workers=num_workers)
    test_loader = TestLoader(test_data, test_label, batch_size, True)
    validate_loader = ValidateLoader(validate_data, validate_label, batch_size, True)

    return train_loader, test_loader, validate_loader, test_data, test_label


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
          11. Nature un-dimensional ResNet\n \
                    "))
    return num


def GetDatapath():
    BASE_DIR = os.getcwd()
    DataPath = os.path.abspath(os.path.join(BASE_DIR, './Data/'))
    mitbd = DataPath + '/mitbd/'
    TrainData_file = '/Train_data.npy'  # you can change it with your training dataset
    TestData_file = '/Test_data.npy'  # you can change it with your testing dataset
    Validate_file = '/Validate_data.npy'  # you can change it with your validate dataset
    Train_Path = DataPath + TrainData_file
    Test_Path = DataPath + TestData_file
    Validate_Path = DataPath + Validate_file

    return DataPath, Train_Path, Test_Path, Validate_Path, mitbd


def gpu_available():
    use_gpu = torch.cuda.is_available()
    return use_gpu
