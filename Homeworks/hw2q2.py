# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:03:30 2022

@author: Pranav
"""
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
def f(x):
    return (np.sin(6*x)**2+3*np.cos(x)**2*np.sin(4*x)**2+1)*np.exp(-x**2/2)

def cdf(x):
    g=integrate.quad(f, -10, x)[0]
    return g

size = 10000
c = 5
values = np.zeros(size)
index = 0
tot = 0
while values[size-1] == 0 :
    entry = norm.rvs()
    unif = uniform.rvs()
    k = f(entry)/(c* norm.pdf(entry))
    if unif < k:
        values[index] = entry
        index = index + 1
    tot = tot + 1

accep_prob = size / tot
k = 1/(c*accep_prob)
print(k)
#0.21932
def empi_cdf(x,n):
    ans = np.repeat(0,n)
    for i in range(n):
        ans[i] = sum( x<= (i+1)/n)  
    return ans 
y = empi_cdf(values, 500)/size

x= np.arange(-10,10,20/500)
plt.plot( x, empi_cdf(values, 500))  
