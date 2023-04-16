# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 08:39:04 2022

@author: Pranav
"""
import numpy as np
from numpy.fft import fft, ifft

def covariance(N1,N2):
    f = abs(N2 - N1)
    return h**(2*H) - 2*f**(2*H) + (abs(h + N2 - N1))**(2*H) + (abs(h + N1 - N2))**(2*H)
    
def cov(V1):
    ans = np.repeat(0., len(V1) + 1)
    for i in range(len(V1) ):
        ans[i] = covariance(V1[0], V1[i])
    return ans
        

H = 0.9
h = 0.002
T0 = 0
T1 = 1
N = 50
T = np.arange(T0,T1,h)
temp = cov(T)
temp2 = np.flip(temp[2:-1])
alpha = np.concatenate((temp,temp2))
lambda_variable = fft(alpha)

Y = np.random.normal(size = (N, 2)).view(np.complex128)
diag_sqrt = alpha**0.5
X_tilda = ifft((2*N)**(0.5)*diag_sqrt*Y)
X_1 = np.real(X_tilda)[0:N]


