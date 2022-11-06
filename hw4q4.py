# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 08:13:00 2022

@author: Pranav
"""

from scipy.stats import poisson, uniform
import matplotlib.pyplot as plt
import numpy as np
N2 = 20
N1 = 1
lamda = 0.1
X = poisson.rvs(mu = lamda*(N2 - N1))
T = uniform.rvs(loc = N1, scale = N2, size = X)
plt.plot(np.arange(N1,N2,(N2 - N1)/X),sorted(T))
