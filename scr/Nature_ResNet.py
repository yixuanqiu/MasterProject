import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding, downsample=None):
        super(BasicBlock, self).__init__()
        self.padding = padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=7, stride=4, bias=False)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(0.8)
        self.bn = nn.BatchNorm1d(out_channels)
        self.maxpooling = nn.MaxPool1d(kernel_size=4, stride=4, padding=self.padding)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # first layer
        out = self.conv1(x)
        out = self.dropout(F.relu(self.bn(out)))
        out = self.conv2(out)

        # skipping layer
        y = self.maxpooling(y)
        y = self.conv3(y)
        out += y

        # second layer
        z = out
        z = self.relu(z)
        z = self.dropout(z)
        z += out

        return z, out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.ResBlk1 = BasicBlock(64, 128, padding=0)
        self.ResBlk2 = BasicBlock(128, 192, padding=1)
        self.ResBlk3 = BasicBlock(192, 256, padding=2)
        self.ResBlk4 = BasicBlock(256, 320, padding=2)
        self.sigma = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(320 * 8, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        y = x
        x, y = self.ResBlk1(x, y)
        x, y = self.ResBlk2(x, y)
        x, y = self.ResBlk3(x, y)
        x, _ = self.ResBlk4(x, y)
        x = x.view(x.size(0), -1)
        x = self.sigma(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x
