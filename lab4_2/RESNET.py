#!/usr/bin/env python3
import torch.nn as nn
from torchvision import transforms,models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class ResNet18(nn.Module):
    def __init__(self, num_class, feature_extract, use_pretrained):
        super(ResNet18,self).__init__()
        self.model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(self.model, feature_extract)
        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,5)
        # self.activate = nn.LeakyReLU()
        # self.fc = nn.Linear(256, num_class)

    def forward(self,x):
        out = self.model(x)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_class, feature_extract, use_pretrained):
        super(ResNet50,self).__init__()
        self.model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(self.model, feature_extract)
        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)
        self.activate = nn.LeakyReLU()
        
    def forward(self,x):
        out=self.model(x)
        return out