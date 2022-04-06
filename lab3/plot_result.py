#!/usr/bin/env python3

import numpy as np
import argparse
import matplotlib.pyplot as plt

def read(file):
    episode = []
    score = []
    with open(file) as f:
        Lines = f.readlines()
        for line in Lines:
            ep, sc = line.split(",")
            episode.append((int)(ep.split(":")[1]))
            score.append((int)(sc.split(":")[1]))

    return np.array(episode), np.array(score)     

def plt_result(epoches, score, file):

    plt.title("result", fontsize = 18)

    #plt.plot(episode[0:100000:2000], score[0:100000:2000])
    plt.plot(episode, score)
    plt.xlim(0,100000)
    plt.ylim(0,200000)

    plt.xlabel("episode")
    plt.ylabel("score")

    plt.savefig('{0}.png'.format(file))
    plt.show()

if __name__ == "__main__":

    episode, score = read("./record_score/score.txt")
    plt_result(episode, score, "result")