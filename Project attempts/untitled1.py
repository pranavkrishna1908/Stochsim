# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 17:04:47 2022

@author: Pranav
"""
import numpy as np

#run after getting the son
#embedding
def imbed(X):
    answer = np.zeros((10,10))
    answer= X[:11,:11]
    return answer

Y = np.array([embed(Xi) for Xi in X])
e_11 = np.array([0,0,0,0,0,0,0,0,0,0,1])

vects = np.array([Xi@e_11 for Xi in X])  
  
vect_norms = np.array([np.linalg.norm(v) for v in vects])

plt.subplots()
plt.hist(vect_norms)
plt.show()
