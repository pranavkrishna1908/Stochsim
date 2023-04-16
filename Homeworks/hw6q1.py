# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 08:18:53 2022

@author: Pranav
"""
import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt

a = 2
def A(point):
    if point[1] > a:
        if point[0] >a:
            return(1)
    return(0)


iterations = 10000
sigma = np.array([[4,-1],[-1,4]])
L = np.array([[2,0], [-0.5, np.sqrt(15/4)]])

def Crude(iterations):
    temp = 0
    for i in range(iterations):
        idvec = norm.rvs(size = 2)
        vec = np.matmul(L, idvec)
        temp = A(vec) + temp
    x_bar = temp/iterations
    sigma_sq = ((temp - x_bar)**2)/iterations
    return (x_bar - 1.96*sigma_sq/np.sqrt(iterations), x_bar + 1.96*sigma_sq/np.sqrt(iterations))

def importance_sampling_mean(iterations):
    arraything = np.zeros(iterations)
    for i in range(iterations):
        idvec = norm.rvs(size = 2)
        vec = np.matmul(L, idvec) + (a,a)
        arraything[i] = A(vec)*multivariate_normal.pdf(vec, cov = sigma)/ multivariate_normal.pdf(vec,mean = (a,a), cov = sigma)
    estimator = sum(arraything) / iterations
    sigma_sq = sum((arraything - estimator)**2)/iterations
    print(sigma_sq)
    #print(estimator - 1.96*sigma_sq/np.sqrt(iterations), estimator + 1.96*sigma_sq/np.sqrt(iterations))
    
def importance_sampling_cov(delta, iterations):
    arraything = np.zeros(iterations)
    for i in range(iterations):
        idvec = norm.rvs(size = 2)
        vec = np.sqrt(delta)*np.matmul(L, idvec) + (a,a)
        arraything[i] = A(vec)*multivariate_normal.pdf(vec, cov = sigma)/ multivariate_normal.pdf(vec,mean = (a,a), cov = delta * sigma)
    estimator = sum(arraything) / iterations
    sigma_sq = sum((arraything - estimator)**2)/iterations
    return(sigma_sq)
    #print(estimator - 1.96*sigma_sq/np.sqrt(iterations), estimator + 1.96*sigma_sq/np.sqrt(iterations))
importance_sampling_cov(1, iterations)
importance_sampling_mean(iterations)
thing = np.repeat(0,100)

for i in range(100):
    thing[i] = importance_sampling_cov( (1 +i)/20, iterations)
plt.plot(np.log(thing))
