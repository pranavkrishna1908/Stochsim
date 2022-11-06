# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:30:32 2022

@author: Pranav
"""

from scipy.stats import norm, uniform, expon
import math as mt
import numpy as np

num = 100
X = norm.rvs(loc = 0, scale = 1, size = num)
Y = norm.rvs(loc = 0, scale = 1, size = num)
V1 = X/Y

uniform_values = uniform.rvs(loc = 0, scale = 1, size = num)
desired_vales = np.tan(mt.pi*(uniform_values - 0.5))

#Shift after getting the standard cauchy distribution