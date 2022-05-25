#!/usr/bin/env python3
import torch.nn as nn

class EEG(nn.Module):
	def __init__(self,act_func='ELU'):
		super(EEG, self).__init__()
		if act_func == 'ELU': 
			self.active_function = nn.ELU()
		if act_func == 'LeakyReLU': 
			self.active_function = nn.LeakyReLU()
		if act_func == 'ReLU': 
			self.active_function = nn.ReLU()

		self.pipe0 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,51), stride=(1,1),padding=(0,25), bias=False),
			nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
			nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
		)
		self.pipe1 = nn.Sequential(
			nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
			nn.Dropout(p=0.25),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,15),stride=(1,1),padding=(0,7), bias=False),
			nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
		)
		self.pipe2 = nn.Sequential(
			nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
			nn.Dropout(p=0.25),
			nn.Flatten(),
			nn.Linear(in_features=736,out_features=2,bias=True)
		)
	def forward(self,x):
		x = self.pipe0(x)
		x = self.active_function(x)
		x = self.pipe1(x)
		x = self.active_function(x)
		x = self.pipe2(x)
		return x