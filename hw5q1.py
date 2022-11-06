# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 08:23:44 2022

@author: Pranav
"""

import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

def iteration():
    a = uniform.rvs(loc= -1, scale = 2, size = 2)
    return (sum(a*a), sum(a))
np.random.seed(98)
N = np.array([1,2,3,4,5])
answers = np.repeat(0., 20)
means = np.repeat(0., 5)
for i in N:
    sumsofar = 0
    sqsumsofar = 0
    num = 10**i
    for j in range(num):
        temp = iteration()
        sumsofar =  temp[1] + sumsofar
        sqsumsofar = temp[0] + sqsumsofar
    mean = sumsofar/num
    means[i-1] = mean
    variance = sqsumsofar/num - mean**2
    answers[2*i] = mean - np.sqrt(variance)*1.95/np.sqrt(num)
    answers[2*i + 1] = mean + np.sqrt(variance)*1.95/np.sqrt(num)
vector = np.repeat(0., 5)
for i in range(5):
    vector[i] = answers[2*i + 3] - answers[2*i + 1]
plt.plot(range(5), abs((vector)/means))
        