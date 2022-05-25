#!/usr/bin/python3
import numpy as np
import random
import os
import copy
import itertools
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import matplotlib.pyplot as plt
from RESNET import ResNet18,ResNet50
from dataloader import RetinopathyLoader



class Trainer():
    def __init__(self,model,param,optimizer):
        self.model = model
        self.param = param
        self.optimizer = optimizer

        self.num_class = 5

        self.trainset = RetinopathyLoader('/home/ncrl/ws/dlp_ws/dl_hw/lab4_2/data',"train")
        self.testset = RetinopathyLoader('/home/ncrl/ws/dlp_ws/dl_hw/lab4_2/data',"test")

        self.trainloader = DataLoader(dataset=self.trainset, batch_size=self.param['Batch_size'], shuffle=True, num_workers=2)
        self.testloader = DataLoader(dataset=self.testset, batch_size=self.param['Batch_size'], shuffle=True, num_workers=2)
        
        self.weight_loss = torch.from_numpy(self.trainset.weight_loss).float()
        
        self.criterion = nn.CrossEntropyLoss(weight=self.weight_loss) if self.param['weight_loss'] else nn.CrossEntropyLoss() 
        
        self.model = self.model.cuda()
        self.criterion = self.criterion.cuda()
        self.loss_record = []

        # crate folder
        self.weight_path = os.path.join(os.getcwd(),"weight")
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)

        self.picture_path = os.path.join(os.getcwd(),"picture")
        if not os.path.exists(self.picture_path):
            os.makedirs(self.picture_path)

        self.record_path = os.path.join(os.getcwd(),"record")
        if not os.path.exists(self.record_path):
            os.makedirs(self.record_path)

        text = "with_pretrained" if param['use_pretrained'] else "without_pretrained"
        self.file_name ="{0}_{1}_{2}_{3}_{4}".format(param['Model_name'], text, param['Batch_size'], param['Epochs'],  param['Learning_Rate'])

    def train(self):
        train_loss = 0.0
        tbar = tqdm(self.trainloader)
        self.model.train()

        for i, (data, label) in enumerate(tbar):
            data, label = Variable(data),Variable(label)
            data, label = data.cuda(), label.cuda()
            prediction = self.model(data)
            loss = self.criterion(prediction, label.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            self.optimizer.zero_grad()
            tbar.set_description('Train loss: {0:.6f}'.format(train_loss / (i + 1)))
        
        return train_loss / (i + 1)

    def evaluate(self, d):
        correct = 0.0
        total = 0.0
        tbar = tqdm(d)
        self.model.eval()
        for i, (data, label) in enumerate(tbar):
            data, label = Variable(data),Variable(label)
            data, label = data.cuda(), label.cuda()
                
            with torch.no_grad():
                prediction = self.model(data)
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += np.sum(np.squeeze(pred.eq(label.data.view_as(pred))).cpu().numpy())
                total += data.size(0)

            text = "Train accuracy" if d == self.trainloader else "Test accuracy"
            tbar.set_description('{0}: {1:2.1f}% '.format(text, 100. * correct / total))

        return 100.0 * correct / total

    def process(self):
        best_model_weight = None
        best_accuracy = 0.0

        for epoch in range(self.param['Epochs']):
            loss = self.train()
            self.loss_record.append(loss)
            tr_ac = self.evaluate(self.trainloader)
            te_ac = self.evaluate(self.testloader)
            self.record(epoch+1, loss, tr_ac, te_ac)
            if(te_ac > best_accuracy):
                best_accuracy = te_ac
                best_model_weight = copy.deepcopy(self.model.state_dict())

        # save the best model 
        torch.save(best_model_weight, os.path.join(self.weight_path, self.file_name + '.pkl'))
        # plot 
        self.plot_confusion_matrix(5)
        self.plt_lr_cur()

    def load(self):
        # for epoch in range(self.param['Epochs']):
        #     te_ac = self.evaluate(self.testloader)
        te_ac = self.evaluate(self.testloader)


    def plt_lr_cur(self):
        plt.figure(2)
        plt.title("learning curve", fontsize = 18)
        ep = [x for x in range(1, self.param['Epochs'] + 1)]
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(ep, self.loss_record)
        plt.savefig(os.path.join(self.picture_path, self.file_name + '_learning_curve.png'))
        #plt.show()

    def record(self, epochs, loss, tr_ac, te_ac):
        file_path = os.path.join(self.record_path, self.file_name + '.txt')
        with open(file_path, "a") as f:
            f.writelines("Epochs : {0}, train loss : {1:.6f}, train accurancy : {2:.2f}, test accurancy : {3:.2f}\n".format(epochs, loss, tr_ac, te_ac))

    def plot_confusion_matrix(self, num_class):
        matrix = np.zeros((self.num_class,self.num_class))
        self.model.eval()
        for i, (data, label) in enumerate(self.testloader):
            data, label = Variable(data),Variable(label)
            data, label = data.cuda(), label.cuda()
                
            with torch.no_grad():
                prediction = self.model(data)
                pred = prediction.data.max(1, keepdim=True)[1]

                ground_truth = pred.cpu().numpy().flatten()
                actual = label.cpu().numpy().astype('int')

                for j in range(len(ground_truth)):
                    matrix[actual[j]][ground_truth[j]] += 1
                
        for i in range(self.num_class):
            matrix[i,:] /=  sum(matrix[i,:])

        plt.figure(1)
        plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Normalized confusion matrix")
        plt.colorbar()

        thresh = np.max(matrix) / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

        tick_marks = np.arange(self.num_class)
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(self.picture_path, self.file_name + '_confusion_matrix.png'))

if __name__ == '__main__':

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper param
    hyper_param_18 = {
        'Model_name' : 'ResNet18', 
        'Learning_Rate' : 0.001, 
        'Batch_size'  : 16, 
        'momentum' : 0.9, 
        'weight_decay' : 5e-4, 
        'Epochs' : 10 ,
        'num_class' : 5,
        'use_pretrained' : True,
        'weight_loss' : False
    }

    hyper_param_50 = {
        'Model_name' : 'ResNet50', 
        'Learning_Rate' : 0.001,
        'Batch_size'  : 8,
        'momentum' : 0.9,
        'weight_decay' : 5e-4,
        'Epochs' : 5,
        'num_class' : 5,
        'use_pretrained' : False,
        'weight_loss' : False
    }

    model = ResNet18(num_class=5,feature_extract=False,use_pretrained=hyper_param_18['use_pretrained'])
    PATH = '/home/ncrl/ws/dlp_ws/dl_hw/lab4_2/weight/ResNet18_with_pretrained_16_10_0.0018.pkl'
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper_param_18['Learning_Rate'], momentum=0.9)
    model.load_state_dict(torch.load(PATH))
    model.to('cuda:0')
    Trainer = Trainer(model,hyper_param_18,optimizer)
    Trainer.load()


