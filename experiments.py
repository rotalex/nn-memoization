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

def init_weights_he(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.kaiming_uniform_(m.weight)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0001)

class Experiment:
    def __init__(
        self,
        model,
        train_data,
        test_data,
        optimizer_class,
        criterion,
        device,
        epochs=10000,
        decode=None,
        init_weigths = True,
    ):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.opt = optimizer_class(self.model.parameters(), lr=1e-4)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.95, patience=10,
            verbose=False, threshold=0.01, min_lr=1e-7, eps=1e-08
        )
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        if decode:
            self.decode = decode
        
        if init_weigths:
            self.model.apply(init_weights_he)
    
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
                print("\rEpoch: %5d:" % (e), curr, end='')
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
