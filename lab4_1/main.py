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

i = 0
def train( model, train_data, train_label, optimizer, batchsize):
	global i
	count = 0
	model.train()
	while count<1080:
		data = torch.cuda.FloatTensor( train_data[i:i+batchsize] )
		target = torch.cuda.LongTensor( train_label[i:i+batchsize] )
		optimizer.zero_grad()
		output = model(data)
		loss = nn.CrossEntropyLoss()
		loss = loss(output, target)
		loss.backward()
		optimizer.step()

		i = (i+batchsize)%1080
		count += batchsize

def test(model, test_data, test_label):
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
	return test_loss.item()/1080.0 , correct/1080.0



if __name__ == '__main__':
	torch.manual_seed(1)
	device = torch.device('cuda:0')
	train_data, train_label, test_data, test_label = read_bci_data()
	# Hyper param
	Learning_Rate = 0.001
	Batch_size  = 128
	Opti = 'Adam' # SGD/Adagrad/Adam/RMSprop
	print ('learning rate: '+str(Learning_Rate))
	print ('batch_size: '+str(Batch_size))
	print ("optimizer : "+str(Opti))



	for task in ['EEG', 'DeepConvNet']:
		plt_array_loss = []
		plt_array_accuracy = []
		for act_func in ['ReLU','LeakyReLU', 'ELU']:
			for testset in ['train','test']:
				#print(str(task+'_'+act_func+'_'+testset))
				plt_array_loss_tmp = []
				plt_array_accuracy_tmp = []


				if testset == 'train':
					m_data, m_label = train_data, train_label
				elif testset == 'test':
					m_data, m_label = test_data, test_label

				if task == 'EEG':
					model = EEG(act_func=act_func)
				elif task == 'DeepConvNet':
					model = DeepConvNet(act_func=act_func)


				if Opti == 'SGD':
					optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate, momentum=0.9)
				elif Opti == 'Adagrad':
					optimizer = torch.optim.Adagrad(model.parameters(), lr=Learning_Rate)
				elif Opti == 'RMSprop':
					optimizer = torch.optim.RMSprop(model.parameters(), lr=Learning_Rate, alpha=0.9)
				else : # default Adam
					optimizer = optim.Adam(model.parameters(),lr=Learning_Rate)

				# demo
				PATH = '/home/ncrl/ws/dlp_ws/dl_hw/lab4_1/hw3EEGLeakyReLUtest.pt'
				model.load_state_dict(torch.load(PATH))
				model.to(device)
				m_data, m_label = test_data, test_label


				for epoch in range(601):
					train(model, train_data, train_label, optimizer, batchsize=Batch_size)
					test_loss, correct = test(model, m_data, m_label)
					plt_array_accuracy_tmp.append(correct*100)
					plt_array_loss_tmp.append(test_loss)
					if epoch%200 == 0 or (correct>=0.85 and testset == 'test') : print('epoch= ',epoch,' loss= ',test_loss,' correct= ',correct)

				plt_array_accuracy.append(plt_array_accuracy_tmp)
				plt_array_loss.append(plt_array_loss_tmp)
				torch.save(model.state_dict(), str('hw3'+task+act_func+testset+'.pt'))



		for arr in plt_array_accuracy: plt.plot(arr)
		plt.title(str("Activation Functions comparision ("+task+')'))
		plt.grid()
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy(%)')
		plt.legend(['relu_train', 'relu_test', 'leaky_relu_train', 'leaky_relu_test', 'elu_train', 'elu_test',])
		plt.savefig(str(task+'_accuracy.png'))
		plt.close()
		plt.show()

		for arr in plt_array_loss: plt.plot(arr)
		plt.grid()
		plt.title(str("Learning curve comparision ("+task+')'))
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend(['relu_train', 'relu_test', 'leaky_relu_train', 'leaky_relu_test', 'elu_train', 'elu_test',])
		plt.savefig(str(task+'_loss.png'))
		plt.close()
		plt.show()