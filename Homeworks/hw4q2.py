# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:30:17 2022

@author: Pranav
"""

import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt

a= 0.4
N = 1000
path = np.repeat(0, N)

def step(prev): 
    temp1 = uniform.rvs()
    temp2 = uniform.rvs()
    if temp1 > 2*a:
        return prev
    else:
        if temp2 < 0.5:
            return prev - 1
    return prev + 1

def realisation(path):
    num = len(path)
    ans = np.repeat(0, N)
    for i in range(num - 1):
        ans[i+1] = step(ans[i])
    return ans
realisation(path)

for i in range(20):
    path = np.repeat(0, N)
    path = realisation(path)
    modified_path = path/(2*a)**0.5
    plt.plot(np.arange(0, N,1), modified_path)

def weiner_realisation(path):
    for i in range(len(path) - 1):
        path[i+1] = path[i] + norm.rvs(loc = 0, scale = 1/(N/(2*a)**0.5))
        return path

for i in range(20):
    path = np.repeat(0.,int(N/(2*a)**0.5))
    path = weiner_realisation(path)
    plt.plot(np.arange(0, len(path),1),path)

    

