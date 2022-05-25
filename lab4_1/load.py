from dataloader import read_bci_data

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import matplotlib.pyplot as plt

from Deep_Conv_Net import DeepConvNet
from EEG_NET import EEG




model = EEG(act_func='LeakyReLU')
PATH = '/home/ncrl/ws/dlp_ws/dl_hw/lab4_1/hw3EEGLeakyReLUtest.pt'
train_data, train_label, test_data, test_label = read_bci_data()
model.load_state_dict(torch.load(PATH))
model.to('cuda:0')

train_data, train_label, test_data, test_label = read_bci_data()
model.eval()
data = torch.cuda.FloatTensor( test_data )
target = torch.cuda.LongTensor( test_label )
output = model(data)
loss = nn.CrossEntropyLoss()
test_loss = loss(output, target)  # sum up batch loss
pred = output.argmax(dim=1)  # get the index of the max log-probability
correct = 0
for i,pred_ans in enumerate(pred):
    if pred[i] == target[i]: correct += 1
print(correct/1080.0)

