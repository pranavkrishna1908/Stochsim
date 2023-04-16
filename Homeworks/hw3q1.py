# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:45:24 2022

@author: Pranav
"""

from scipy.stats import norm
import numpy as np
import scipy.linalg
import pprint
import matplotlib.pyplot as plt

A = [[1,2],[2,5]]
N = 10**6
X = norm.rvs(size = N)
Y = norm.rvs(size = N)
T = list()
mu = (2,1)
L = scipy.linalg.cholesky(A, lower=True)
pprint.pprint(L)
for i in range(N):
    temp1 = X[i]
    temp2 = Y[i]
    temp = (temp1, temp2)
    T.append(mu + np.matmul(L, temp))

for i in range(N):
    X[i] = T[i][0]
    Y[i] = T[i][1]
# Big bins
plt.hist2d(X,Y, bins=(50, 50), cmap=plt.cm.jet)
plt.show()
 
# Small bins
plt.hist2d(X,Y, bins=(300, 300), cmap=plt.cm.jet)
plt.show()
#Not full rank
N = 10**6
X = norm.rvs(size = N)
Y = 2*X

# Big bins
plt.hist2d(X,Y, bins=(50, 50), cmap=plt.cm.jet)
plt.show()
 
# Small bins
plt.hist2d(X,Y, bins=(300, 300), cmap=plt.cm.jet)
plt.show()
 
