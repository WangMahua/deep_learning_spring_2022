#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class nn_model():
    def __init__(self):
        self.input_size = 2
        self.output_size = 1
        self.epochs = 10000

        # initialize weight value
        self.w1 = np.random.rand(self.input_size,self.input_size)
        self.w2 = np.random.rand(self.input_size,self.input_size)
        self.w3 = np.random.rand(self.input_size,self.output_size)

        # record mid layer
        self.input_data = []
        self.layer1_data = []
        self.layer2_data = [] 
        self.output_label = []

        # record truth data
        self.truth_data = []

        #SGD
        self.learning_rate = 0.01
        self.grad_w1 = []
        self.grad_w2 = []
        self.grad_w3 = []
        self.grad_1 = []
        self.grad_2 = []
        self.grad_3 = []


    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))

    def derivative_sigmoid(self,x):
        return np.multiply(x, 1.0-x)

    def loss_function(self,y,pred_y):
        return float((np.linalg.norm(y-pred_y)**2)/len(y))

    def network(self,weight,data): 
        temp = np.dot(data, weight)
        temp = self.sigmoid(temp)
        return temp

    def forward_pass(self,data): 
        self.layer1_data = self.network(self.w1,data)
        self.layer2_data = self.network(self.w2,self.layer1_data)
        self.output_label = self.network(self.w3,self.layer2_data)

    def update_w(self):
        self.w1 = self.w1 - self.learning_rate * self.grad_w1
        self.w2 = self.w2 - self.learning_rate * self.grad_w2
        self.w3 = self.w3 - self.learning_rate * self.grad_w3

    def update_grad(self):
        self.grad_3 = self.derivative_sigmoid((self.output_label)*(2*(self.truth_data-self.output_label)))
        print(2*(self.truth_data-self.output_label))
        self.grad_w3 = np.dot(self.layer2_data.T, self.grad_3)
        self.grad_2 = self.derivative_sigmoid(self.layer2_data.T*(np.dot(self.grad_3,self.w3.T)))
        self.grad_w2 = np.dot(self.layer1_data.T, self.grad_2)
        self.grad_1 = self.derivative_sigmoid(self.layer1_data.T*(np.dot(self.grad_2,self.w2.T)))
        self.grad_w1 = np.dot(self.input_data.T, self.grad_1)

    def back_propagation(self): 
        self.update_grad()

    def evaluate(self,y,pred_y):
        print(pred_y)
        correct_task = 0
        for i in range(len(y)):
            if y[i] == pred_y[i]:
                correct_task = correct_task + 1
        accuracy = correct_task / len(y)
        print("accurancy : {0}%, loss : {1:.6f}".format(accurancy, self.loss_function(y, pred_y)))

    def process(self,origin_data,label):
        self.data = origin_data
        self.truth_data = label
        self.output_label = np.ones(len(origin_data))
        for i in range(self.epochs):
            loss_error = self.loss_function(label,self.output_label)
            if  loss_error > 0.01 : 
                self.forward_pass(origin_data)
                self.back_propagation()

            if self.epochs % 500 == 0:
                self.evaluate(label,self.output_label)
        return self.output_label
    

        