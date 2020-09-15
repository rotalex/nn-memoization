from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import scipy
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from functools import partial

import argparse
from copy import deepcopy

parser = argparse.ArgumentParser(
    description='Generate hardcoded neural network for dataset containg 512 entries.'
)

parser.add_argument(
    '--dataset_path',
    default="datasets/1.csv",
    type=str,
    help='path towards csv containing the data'
)

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

class SmallestNNDatasetBitEncodedMinusOne2One(Dataset):
    def __init__(self, df, count=-1):
        super(SmallestNNDatasetBitEncodedMinusOne2One, self).__init__()
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

def minus_one_2_one_convert(two_bytes):
    data_i_cpy = deepcopy(two_bytes)
    data_i_cpy[data_i_cpy < 0] = 0
    return decode(data_i_cpy)

class HardcodedSmallestNN(torch.nn.Module):
    def __init__(self, input_size, dataset):
        super(HardcodedSmallestNN, self).__init__()
        
        data = SmallestNNDatasetBitEncodedMinusOne2One(dataset)
        self.key_recognizer = nn.Conv1d(1, 512, kernel_size=input_size, bias=False)
        self.key2out1 = nn.Linear(512, 1, bias=False)
        self.key2out2 = nn.Linear(512, 1, bias=False)

        for i in range(len(dataset)):
            self.key_recognizer.weight.data[i][0] = data[i][0] / torch.abs(data[i][0]).sum()
            data_i_cpy = deepcopy(data[i][0])
            data_i_cpy[data_i_cpy < 0] = 0

        self.key2out1.weight.data[0] = torch.Tensor(dataset['out1'])
        self.key2out2.weight.data[0] = torch.Tensor(dataset['out2'])

    def forward(self, x):
        key_activation_maps = self.key_recognizer(x).view(-1, 512)
        one_hot = F.one_hot(key_activation_maps.argmax(), num_classes=512).float()
        predicted_out1 = self.key2out1(one_hot)
        predicted_out2 = self.key2out2(one_hot)
        
        return torch.cat((predicted_out1, predicted_out2))

    
if __name__ == "__main__":
    args = parser.parse_args()
    dataset = pd.read_csv(args.dataset_path)
    hnn = HardcodedSmallestNN(16, dataset)
    example = torch.rand(1, 1, 16)
    traced_script_module = torch.jit.trace(hnn.cpu(), example)
    traced_script_module.save(f"model_hardcode_acc1.0_{os.path.basename(args.dataset_path)}.pt")