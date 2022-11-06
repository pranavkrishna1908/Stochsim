# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:35:55 2022

@author: Pranav
"""

from scipy.stats import norm, uniform, expon
import math as mt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

startTime = time.perf_counter()
num = 200

values1 = uniform.rvs(loc =0, scale = 1, size = num)

time1 = time.perf_counter()

sorted(values1)
def unif_order(num):    
    values2 = np.repeat(0., num)
    values2[num-1] = uniform.rvs(loc = 0, scale = 1)
    for i in range(num - 1):
        temp = uniform.rvs(loc = 0, scale = values2[num-1 - i])
        values2[num-2-i] = temp**(1/(num-1-i))*(values2[num-1-i])
    return values2    
    # time2 = time.perf_counter()
    # print(time2 - time1, time1 - startTime)
N = 10000
X = np.repeat(0.,N)
Y = np.repeat(0.,N)
Z = np.repeat(0.,N)
for i in range(N):
    orderstat = unif_order(3)
    X[i] = orderstat[0]
    Y[i] = orderstat[1]
    Z[i] = orderstat[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
