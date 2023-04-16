# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
import math as mt

def ecdf(x, n):
    ecdf = np.zeros(n)
    for i in range(n):
        ecdf[i] = sum(x <= -1 + 4*(i+1)/n)   
    return ecdf
def cdf(x):
    ans = np.zeros(len(x))
    flags1 = [x>2]*np.arange(len(x))
    flags2 = [(x<2) * (x>= 0)]
    flags2 = (np.repeat(1,len(x))-flags2)*np.arange(len(x))
    ans = 1-(2/3)*np.exp(-(x/2))
    ans[flags2] = 0
    ans[flags1] = 1
    return ans
nums = 5000
transformed_values = np.zeros(nums)
uniform_values = uniform.rvs(loc = 0, scale = 1, size = nums)
flags1 = (uniform_values > 1-2/(3*mt.exp(1)))*np.arange(nums)
flags2 = (uniform_values < 1/3)*np.arange(nums)
transformed_values = -2*np.log(1.5*(1-uniform_values))
transformed_values[flags2] = 0
transformed_values[flags1] = 2
desired_values = transformed_values
n = 500
empi_cdf = ecdf(desired_values, n)/nums
plt.plot( np.arange(-1,3,4/n), empi_cdf, 'o')
plt.plot(np.arange(-1,3,4/n), cdf(np.arange(-1,3,4/n)),'r')