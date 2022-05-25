#!/usr/bin/env python3
import torch.nn as nn

class DeepConvNet(nn.Module):
	def __init__(self,act_func='ELU'):
		super(DeepConvNet, self).__init__()
		if act_func == 'ELU': 
			self.active_function = nn.ELU()
		elif act_func == 'LeakyReLU': 
			self.active_function = nn.LeakyReLU()
		elif act_func == 'ReLU': 
			self.active_function = nn.ReLU()

		C, T, N = 2, 750, 2

		self.pipe0 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,5), stride = (1,2)),
			nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(C,1)),
			nn.BatchNorm2d(25, eps=1e-5, momentum=0.1)
		)
		self.pipe1 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,5), stride = (1,2)),
			nn.BatchNorm2d(50, eps=1e-5, momentum=0.1)
		)
		self.pipe2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1,5)),
			nn.BatchNorm2d(100, eps=1e-5, momentum=0.1)
		)
		self.pipe3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1,5)),
			nn.BatchNorm2d(200, eps=1e-5, momentum=0.1)
		)
		self.pipe4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Flatten(),
			nn.Linear(in_features=1600,out_features=2)
		)

	def forward(self,x):
		x = self.pipe0(x)
		x = self.active_function(x)
		x = self.pipe1(x)
		x = self.active_function(x)
		x = self.pipe2(x)
		x = self.active_function(x)
		x = self.pipe3(x)
		x = self.active_function(x)
		x = self.pipe4(x)
		return x
