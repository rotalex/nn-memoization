#I_AM_NOT_TRYING_TRICK_PAWEL.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


from scipy import stats
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


class SmallestNN(torch.nn.Module):
    def __init__(self, input_size, lable_size):
        super().__init__()
        
        nn.ConvTranspose1d(1, 512, 3)
        self.seq = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Sigmoid(),
            torch.nn.Linear(512, 512),
            nn.Sigmoid(),
            torch.nn.Linear(512, 512),
            nn.Sigmoid(),
            torch.nn.Linear(512, lable_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class SmallestNNBitVectV2(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        mdls = []
        for (inp, out) in zip(sizes[:-1], sizes[1:]):
            mdls.append(nn.Linear(inp, out))
            mdls.append(nn.LeakyReLU())
        self.seq = nn.Sequential(*mdls)

    def forward(self, x):
        x = self.seq(x)
        return x

class SmallestNNBitVectV3(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        mdls = []
        for (inp, out) in zip(sizes[:-1], sizes[1:]):
            mdls.append(nn.Linear(inp, out))
            mdls.append(nn.Tanh())
        self.seq = nn.Sequential(*mdls)

    def forward(self, x):
        x = self.seq(x)
        return x

class SmallestCNNBitVect(torch.nn.Module):
    def __init__(self, input_size, label_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(256, 16)
        self.bn1 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = self.conv1(x).view(-1, 256)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class SmallestNNRecurrent(torch.nn.Module):
    def __init__(self, *sizes, layer_type="lstm"):
        super().__init__()
        
        if layer_type == "lstm":
            self.recurrence = nn.LSTM(sizes[0], sizes[1])
        elif layer_type == "gru":
            self.recurrence = nn.GRU(sizes[0], sizes[1])

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        try:
            x, _ = self.recurrence(x)
        except RuntimeError:
            import pdb; pdb.set_trace()
        return x

class SmallestCNNBitVect(torch.nn.Module):
    def __init__(self, input_size, label_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.conv1(x).view(-1, 16)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class SmallestNNCNNBitPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mdls = []
        mdls.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=8))
        mdls.append(nn.ReLU())
        mdls.append(nn.Linear(8, 64))
        mdls.append(nn.ReLU())
        mdls.append(nn.Linear(64, 2))
        mdls.append(nn.ReLU())
        self.seq = nn.Sequential(*mdls)

    def forward(self, x):
        x = self.seq(x)
        return x

    
class SmallestNNBits(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        
        mdls = []
        feature_sizes = sizes[:-1]

        for (inp, out) in zip(feature_sizes[:-1], feature_sizes[1:]):
            mdls.append(nn.Linear(inp, out))
            mdls.append(nn.LeakyReLU())
        
        self.common = nn.Sequential(*mdls)
        feature_out, network_out = sizes[-2], sizes[-1]
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_out, network_out),
                nn.ReLU()
            )
            for _ in range(16) # because we predict 16 bits
        ])

    def forward(self, x):
        common = self.common(x)
        return [head(common) for head in self.heads]


class SmallestNNBitsOneHot(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        
        mdls = []
        feature_sizes = sizes[:-1]

        for (inp, out) in zip(feature_sizes[:-1], feature_sizes[1:]):
            mdls.append(nn.Linear(inp, out))
            mdls.append(nn.ReLU())
            
        mdls.append(nn.Linear(feature_sizes[-1], sizes[-1] * 2))
        mdls.append(nn.ReLU())

        self.layers = nn.Sequential(*mdls)
        
    def forward(self, x):
        logits = self.layers(x)
        return logits.view(-1, 16, 2)
