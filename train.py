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
from tqdm import tqdm
from copy import deepcopy



# Load dataset
dataset_path = "datasets/1.csv"
dataset = pd.read_csv(dataset_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode(x):
    return torch.tensor(
        np.unpackbits(np.array(x, dtype=np.uint8))
    ).float()

def decode(bits):
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().cpu().numpy()


    return torch.tensor(
        np.packbits(np.array(bits, dtype=np.uint8), axis=1)
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


class SmallestNNBitVectV2(torch.nn.Module):
    def __init__(self, *sizes):
        super(SmallestNNBitVectV2, self).__init__()
        mdls = []
        for (inp, out) in zip(sizes[:-1], sizes[1:]):
            mdls.append(nn.Linear(inp, out))
            mdls.append(nn.ReLU())
        self.seq = nn.Sequential(*mdls)

    def forward(self, x):
        x = self.seq(x)
        return x

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)

class Experiment:
    def __init__(
        self,
        model,
        train_data,
        test_data,
        optimizer_class,
        criterion,
        device,
        epochs=1000
    ):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.opt = optimizer_class(self.model.parameters(), lr=1e-4)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.95, patience=10, verbose=False, threshold=0.01, min_lr=1e-7, eps=1e-08)
        self.criterion = criterion
        self.epochs = epochs
        self.device = device

        self.model.apply(init_weights)

    def evaluate(self):
        return 0

    def train(self):
        losses = []
        moment = []
        acc = 0

        for e in tqdm(range(self.epochs)):
            cum_loss = 0
            for batch in self.train_data:
                input, label = batch
                input = input.to(self.device)
                label = label.to(self.device)
                predct = self.model(input)
                loss = self.criterion(predct, label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                cum_loss += loss.item()

            curr = self.evaluate()
            if acc < curr:
                acc = curr
                print("Epoch: %5d acc:" % (e), curr.item(), end='\n')
            if acc >= 0.9999:
                break
            self.sched.step(curr.item())

            moment.append(e)
            losses.append(cum_loss / len(self.train_data))
        self.display_learning(moment, losses)

    def display_learning(self, moment, losses):
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(0, self.epochs, self.epochs * 0.025))
        ax.set_yticks(np.arange(0, 2., 0.25))
        plt.scatter(moment, losses)
        plt.grid()
        plt.show()

class BinaryReprExperiment(Experiment):
    def __init__(
        self,
        model = SmallestNNBitVectV2(16, 256 + 64, 256 + 64, 16).to(device),
        train_data = DataLoader(SmallestNNDatasetBitEncodedMinusOne2One(dataset), batch_size=1, shuffle=True),
        test_data = DataLoader(SmallestNNDatasetBitEncodedMinusOne2One(dataset), batch_size=1, shuffle=True),
        optimizer_class = torch.optim.Adam,
        criterion = torch.nn.MSELoss(),
        device = device,
        epochs = 10000
    ):
        super(BinaryReprExperiment, self).__init__(
            model, train_data, test_data, optimizer_class, criterion, device, epochs
        )

    def evaluate(self):
        T = torch.Tensor([0.5]).to(device)  # threshold
        correct = 0
        for batch in self.test_data:
            input, label = batch # human_understandable_*
            input = input.to(device).unsqueeze(0)
            label = label.to(device)
            predct = self.model(input)[0]
            predct = (predct > T).float() * 1
            predct = decode(predct)
            label = decode(label)
            correct += (predct == label).float().sum() / 2.0

        return (correct.float() / len(self.test_data))

trn = BinaryReprExperiment()
trn.train()

model = deepcopy(trn.model)
example = torch.rand(1, 1, 16)
traced_script_module = torch.jit.trace(model.cpu(), example)
traced_script_module.save("model_acc1.0_dataset1.pt")
