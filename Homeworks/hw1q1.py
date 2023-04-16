# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:49:16 2022

@author: Pranav
"""

from scipy.stats import uniform, chi2
import numpy as np
import matplotlib.pyplot as plt

def iid_seq_func(num):
    iid_seq = uniform.rvs(0,1,size = num)
    return iid_seq
def empi_cdf(observations, num):
    ans= np.repeat(0, num)
    for i in range(num):
        ans[i] = sum(observations < (i)/num)
    return ans/len(observations)
def quantile_func(observations):
    observations.sort()
    messy = len(observations)
    return observations[range(0,messy,int(0.1*messy))]
def empi_cdf_real(observations,x):
    return sum(observations < x)/len(observations)

simulations = 400
sample = iid_seq_func(simulations)
thingy = empi_cdf(sample, 20) 
plt.plot(np.arange(0,1,1/20),np.arange(0,1,1/20),'r', np.arange(0,1,1/20), thingy)
quantiles = quantile_func(thingy)
plt.plot(np.arange(0,1,0.1),quantiles)
ans = np.repeat(0, simulations)
for i in np.arange(simulations):
    ans [i] = empi_cdf_real(sample, sample[i])
test_stat = (max(abs(sample - ans)))
ALPHAS = np.array([1.07, 1.21 ,1.34 ,1.61])
sum(ALPHAS > test_stat)
    
def chi_sq_test(observations, alpha):
    intervals = 10
    O_i = np.repeat(0,intervals)
    for i in range(intervals):
        O_i[i] = sum((observations > i/simulations) * (observations < (i+1)/simulations))
    p = 1/intervals
    test_stat = sum(((O_i - simulations*p)**2))/(simulations*p)
    print(test_stat)
    print(chi2.isf(alpha, df = simulations - 1))
    return test_stat < chi2.isf(1-alpha, df = simulations - 1)
print(chi_sq_test(sample, 0.95) )
def serial_test(observations, alpha):
    elements = list()
    for i in range(0, simulations, 2):
        elements.append((observations[i], observations[i+1]))
    n1  = 0
    n2 = 0
    n3 = 0
    n4 = 0
    for a,b in elements:
        if a<0.5:
            if b< 0.5:
                n1 = n1 + 1
            else:
                n2 = n2 + 1
        else:
            if b < 0.5:
                n3 = n3+1
            else:
                n4 = n4 + 1
    E_i = 0.125*simulations
    print(n1,n2,n3,n4)

    test_stat = ((n1 - E_i)**2 + (n2 - E_i)**2 + (n3 - E_i)**2 + (n4 - E_i)**2)/E_i
    print(test_stat)
    return test_stat < chi2.isf(1-alpha, df = 3)   
serial_test(observations = sample, alpha = 0.95)


