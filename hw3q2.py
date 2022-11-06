# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 22:09:18 2022

@author: Pranav
"""
import itertools
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
plt.figure(0)
N= 51
rho = 1/200
t = np.arange(0,1,1/N)
mu = np.sin(t)
Sigma = np.empty([N, N])
for i, j in itertools.product(range(N), range(N)):
    Sigma[i, j] = np.exp(-abs(i-j)/rho)
L = np.linalg.cholesky(Sigma)
Z = norm.rvs(size = N)
T = mu + np.matmul(L, Z)
plt.plot(t,T)
plt.figure(1)
tdash = np.repeat(0., N-1)
for i in range(N-1):
    tdash[i] = 0.5*(t[i] + t[i+1])
Sigma_zz = Sigma
for i, j in itertools.product(range(N-1), range(N-1)):
    Sigma_zz[i, j] = np.exp(-abs(tdash[i]-tdash[j])/rho)
Sigma_yz = np.empty(([N,N-1]))
for i, j in itertools.product(range(N), range(N-1)):
    Sigma_yz[i, j] = np.exp(-abs(t[i]-tdash[j])/rho)
t_final = np.concatenate((t,tdash))
mu = np.sin(t_final)
N1 = 2*N - 1
Sigma = np.empty([N1, N1])
for i, j in itertools.product(range(N1), range(N1)):
    Sigma[i, j] = np.exp(-abs(t_final[i]-t_final[j])/rho)
L = np.linalg.cholesky(Sigma)
Z = norm.rvs(size = N1)
T = mu + np.matmul(L, Z)
Y = T[::2]
Z = T[1::2]
Y = Y + np.matmul(Sigma_yz, np.matmul(np.linalg.inv(Sigma_zz),Z - mu[1::2]))
temp = np.argsort(t_final)
appa = np.concatenate((Z,Y))
T_final = np.zeros(len(t_final))
t_final = np.sort(t_final)
for item in range(len(T_final)):
    T_final[item] = appa[temp[item]]
plt.plot(t_final, T_final)