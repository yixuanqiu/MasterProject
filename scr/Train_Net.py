#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:30:25 2020

@author: andyq
"""
import torch


def Train(Epoch, model, GPU_device, criterion, optimizer, train_loader, test_loader):
    running_loss = 0.0
    for epoch in range(Epoch):
        for batch_idx, data in enumerate(train_loader):
            inputs, target = data
            if GPU_device:
                inputs = inputs.cuda()
                target = target.cuda()

            outputs = model(inputs)

            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss))
                running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, target = data
                if GPU_device:
                    inputs = inputs.cuda()
                    target = target.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        acc = 100 * correct / total
        print(('Accuracy on test set: %d %% [%d  /  %d]' % (acc, correct, total)))


def Train_GRUAttention(Epoch, encoder, decoder,
                       GPU_device, criterion, encoder_optimizer,
                       decoder_optimizer, train_loader, test_loader):
    running_loss = 0.0
    for epoch in range(Epoch):
        for batch_idx, data in enumerate(train_loader):
            inputs, target = data
            if GPU_device:
                inputs = inputs.cuda()
                target = target.cuda()

            encoder_outputs, encoder_hidden = encoder(inputs)
            decoder_hidden = encoder_hidden
            decoder_output, decoder_hidden = decoder(inputs, decoder_hidden, encoder_outputs)

            loss = criterion(decoder_output, target)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss)))
                running_loss = 0.0

            correct = 0
            total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, target = data
                if GPU_device == True:
                    inputs = inputs.cuda()
                    target = target.cuda()

                encoder_outputs, encoder_hidden = encoder(inputs)
                decoder_hidden = encoder_hidden
                decoder_output, decoder_hidden = decoder(inputs, decoder_hidden, encoder_outputs)
                # top_n, top_i = decoder_output.data.topk(1)
                _, predicted = torch.max(decoder_output, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        acc = 100 * correct / total
        print(('Accuracy on test set: %d %% [%d  /  %d]' % (acc, correct, total)))
