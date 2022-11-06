# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:38:43 2022

@author: Pranav
"""
from scipy.stats import norm, uniform, expon
import math as mt
import numpy as np
import time

nums = 100
times = np.zeros(3)
times[0] = time.perf_counter()

def laplace(x):
    return 0.5*mt.exp(abs(x))

def laplace_rv():
    n = expon.rvs()
    p = uniform.rvs(1)
    p = (p > 0.5)
    return p*n - (1-p)*n

c = mt.sqrt(2*mt.exp(1)/mt.pi)
values = np.zeros(nums)
index = 0
tot = 0
while(values[nums-1] == 0):
    entry = laplace_rv()
    unif = uniform.rvs()
    k = norm.pdf(entry)/(c*laplace(entry) )
    if unif < k:
        values[index] = entry
        index = index + 1
    tot = tot + 1
times[1] = time.perf_counter()

values2 = np.zeros(nums)
r_sqrd = expon.rvs(loc = 0, scale = 0.5, size = int(nums/2))
theta = uniform.rvs(scale = 2*mt.pi, size = int(nums/2))
values2[0:int(nums/2)] = np.sqrt(r_sqrd)*np.sin(theta)
values2[int(nums/2):nums] = np.sqrt(r_sqrd)*np.cos(theta)
times[2] = time.perf_counter()
times[2] = times[2] - times[1]
times[1] = times[1] - times[0]
print(times[1:]/100)
#[7.610e-04 1.241e-06]