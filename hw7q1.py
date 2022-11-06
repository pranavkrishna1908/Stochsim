# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 08:23:07 2022

@author: Pranav
"""

import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

def next_iterate(state, alpha):
    x, w = state
    a = uniform.rvs()
    if a < alpha:
        return (x - 1, w/(2*alpha))
    return (x + 1, w/(2*(1-alpha)))
def one_run(alpha):
    realisation = list()
    realisation.append((0,1))
    for i in range(10):
        pres = realisation[-1]
        later = next_iterate(pres, alpha)
        realisation.append(later)
        if later[0] == 4:
            break
    final = realisation[-1][0] == 4
    return (final, realisation[-1][1])
def many_runs(num,alpha):
    temp = 0
    for i in range(num):
        temp2 = one_run(alpha)
        temp = temp + temp2[0] * temp2[1]
    return temp/num
def alphas():
    alphas = np.arange(0.1,0.5, 0.05)
    num = len(alphas)
    means = np.zeros(num)
    sds = np.zeros(num)
    for i in range(num):
        temp = np.zeros(5)
        alpha = alphas[i]
        for j in range(5):
            temp[j] = many_runs(500,alpha)
        means[i] = np.mean(temp)
        sds[i] = np.std(temp)
    plt.plot(alphas, means - sds)
    plt.plot(alphas, means + sds)
    plt.plot(alphas, means)
alphas()