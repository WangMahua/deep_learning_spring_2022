#!/usr/bin/env python3

import numpy as np
import argparse
import matplotlib.pyplot as plt

def read(file):
    train_accurancy = []
    test_accurancy = []
    with open(file) as f:
        Lines = f.readlines()
        for line in Lines:
            _, _, tr, te = line.split(",")
            train_accurancy.append((float)(tr.split(":")[1]))
            test_accurancy.append((float)(te.split(":")[1]))

    return np.array(train_accurancy), np.array(test_accurancy)        
    
def plt_result(with_pretrained, without_pretrained, title):

    plt.title("Result comparison {0}".format(title), fontsize = 18)

    w_train, w_test = read(with_pretrained)
    wo_train, wo_test = read(without_pretrained)

    e = [x for x in range(1,11)]

    plt.plot(e, w_train, label="Train(0.0015)")
    plt.plot(e, w_test, label="Test(0.0015)")
    plt.plot(e, wo_train, label="Train(0.0018)")
    plt.plot(e, wo_test, label="Test(0.0018)")

    plt.legend(loc='upper left')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy(%)")

    plt.xticks(e)

    plt.savefig('{0}.png'.format(title))
    plt.show()

def create_argparse():
    parser = argparse.ArgumentParser(prog="DLP homework 4", description='This code will show with pretrained and without pretrained result with matpltlib')

    parser.add_argument("with_pretrained_file", type=str, default="none", help="input txt file with pretrained result")
    parser.add_argument("without_pretrained_file", type=str, default="none", help="input txt file without pretrained result")
    parser.add_argument("title", type=str, default="none", help="save title in the figure")

    return parser

if __name__ == "__main__":
    with_pretrained_file = '/home/ncrl/ws/dlp_ws/dl_hw/lab4_2/record/old/ResNet18_with_pretrained_16_10_0.0015.txt'
    without_pretrained_file = '/home/ncrl/ws/dlp_ws/dl_hw/lab4_2/record/old/ResNet18_with_pretrained_16_10_0.0018.txt'
    title = ' ResNet18_with_pretrain_8_5 different lr'

    plt_result(with_pretrained_file, without_pretrained_file, title)