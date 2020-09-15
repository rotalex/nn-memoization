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



def encode(x):
    return torch.tensor(
        np.unpackbits(np.array(x, dtype=np.uint8))
    ).float()

def decode(bits, axis=1):
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().cpu().numpy()
    
    
    return torch.tensor(
        np.packbits(np.array(bits, dtype=np.uint8), axis=0)
    )

def hex2dec(arr):
    arr = np.concatenate((np.zeros_like(arr), arr), axis=1)
    return decode(arr).view(-1, 4)

def dec2hex(arr):
    return torch.FloatTensor(
        np.split(
            np.unpackbits(
                np.array(arr * 16.0, dtype=np.uint8),
                axis=1
            ),
            2,
            axis=1
        )[1::2]
    ).reshape((1, -1))[0]

class SmallestNNDatasetUnscaled(Dataset):
    def __init__(self, df, count=-1):
        super(SmallestNNDatasetUnscaled, self).__init__()
        self.count = count
        self.data_frame = df

    def __len__(self):
        if self.count < 0:
            return len(self.data_frame)
        return self.count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data_frame.iloc[idx]
        input = torch.FloatTensor([row['in1'], row['in2']])
        label = torch.FloatTensor([row['out1'], row['out2']])
        return input, label
    
class SmallestNNDataset(Dataset):
    def __init__(self, df, count=-1):
        super(SmallestNNDataset, self).__init__()
        self.count = count
        self.data_frame = df

    def __len__(self):
        if self.count < 0:
            return len(self.data_frame)
        return self.count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data_frame.iloc[idx]
        input = torch.FloatTensor([row['in1'], row['in2'] / 255])
        label = torch.FloatTensor([row['out1'] / 255, row['out2'] / 255])
        return input, label

class SmallestNNDatasetBitEncoded(Dataset):
    def __init__(self, df, count=-1):
        super(SmallestNNDatasetBitEncoded, self).__init__()
        self.count = count
        self.data = {}
        for idx in range(len(df)):
            row = df.iloc[idx]
            input = encode([row['in1'], row['in2']])
            label = encode([row['out1'], row['out2']])
            self.data[idx] = [input, label]

    def __len__(self):
        if self.count < 0:
            return len(self.data)
        return self.count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]

class SmallestNNDatasetBitEncodedMinusOne2One(Dataset):
    def __init__(self, df, count=-1):
        super().__init__()
        self.count = count
        self.data = {}
        for idx in range(len(df)):
            row = df.iloc[idx]
            input = encode([row['in1'], row['in2']])
            label = encode([row['out1'], row['out2']])
            
            input[input < 0.01] = -1.
            self.data[idx] = [input, label]

    def __len__(self):
        if self.count < 0:
            return len(self.data)
        return self.count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]

class SmallestNNDatasetBitEncodeNormalized2BitOneHot(Dataset):
    def __init__(self, df, count=-1):
        super().__init__()
        self.count = count
        self.data = {}
        for idx in range(len(df)):
            row = df.iloc[idx]
            input = encode([row['in1'], row['in2']])
            label = encode([row['out1'], row['out2']]).long().unsqueeze(1)
            
            ## we are savages and use pre-historic pytorch
            label_one_hot = torch.FloatTensor(label.shape[0], 2)
            label_one_hot.zero_()
            label_one_hot.scatter_(1, label, 1)
        
            self.data[idx] = [input, label_one_hot]

    def __len__(self):
        if self.count < 0:
            return len(self.data)
        return self.count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]

class SmallestNNDatasetBitEncodedBaseK(SmallestNNDatasetBitEncoded):
    def __init__(self, df, count=-1):
        super(SmallestNNDatasetBitEncodedBaseK, self).__init__(df, count)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        inp = hex2dec(np.array(list(arr.numpy() for arr in np.split(data[0], 4))))
        out = hex2dec(np.array(list(arr.numpy() for arr in np.split(data[1], 4))))
        inp = inp.float() / 16.0
        out = out.float() / 16.0
        return inp, out

class SmallestNNDatasetBitEncodedToKthBit(Dataset):
    def __init__(self, dataset, kbit, count=-1):
        super(SmallestNNDatasetBitEncodedToKthBit, self).__init__()
        self.dataset = dataset
        self.kthbit = kbit

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inp, out = self.dataset[idx]
        out_cls = np.zeros(2)
        if out[self.kthbit]:
            out_cls[1] = 1.0
        else:
            out_cls[0] = 1.0
            
        return inp, out_cls.astype(np.float)
    
class SmallestNNDatasetBitListEncoding(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inp, out = self.dataset[idx]
        target_bits = []

        for k in range(16):
            target_bit_one_hot = np.zeros(2)
            if out[k]:
                target_bit_one_hot[1] = 1.0
            else:
                target_bit_one_hot[0] = 1.0
            target_bits.append(target_bit_one_hot)

        target_bits = np.array(target_bits)

        return inp, target_bits.astype(np.float)