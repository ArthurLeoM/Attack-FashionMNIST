import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import random
import numpy as np

class cnn_model(nn.Module):
    def __init__(self, input_dim=1, output_dim=10):
        super(cnn_model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k1 = 24
        self.k2 = 48
        self.k3 = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.k1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.k1),
            nn.ReLU(),
        )
        self.conv1_res = nn.Sequential(
            nn.Conv2d(self.k1, self.k1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.k1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.k1, self.k2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(self.k2),
            nn.ReLU(),
        )
        self.conv2_res = nn.Sequential(
            nn.Conv2d(self.k2, self.k2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.k2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.k2, self.k3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.k3),
            nn.ReLU(),
        )
        self.avgpool = nn.AvgPool2d(7)
        #self.fc_layer = nn.Linear(self.k3, self.hidden_dim)
        self.pred_layer = nn.Linear(self.k3, self.output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)                                   # b*1*28*28 -> b*24*28*28
        x = self.conv1_res(x) + x
        x = self.conv2(x)                                   # b*24*28*28-> b*48*14*14
        x = self.conv2_res(x) + x
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)                          # Flatten
        y_pred = self.pred_layer(self.dropout(x))           # b*hidden -> b*10
        y_pred = self.softmax(y_pred)
        return y_pred

