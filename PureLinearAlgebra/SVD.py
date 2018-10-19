# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:58:20 2018

@author: Anthony
"""

import numpy as np
rows = 7
columns = 7 
data = np.zeros((rows, columns))
#Randomly initialize the data matrix
for i in range(rows):
    for j in range(columns):
        data[i][j] = np.random.randint(2*rows*columns) 

VPrime = np.matmul(data.T, data)
UPrime = np.matmul(data, data.T)
w, V = np.linalg.eigh(VPrime)
ww, U = np.linalg.eigh(UPrime)
singularVals = np.sqrt(np.abs(w))
singularVals = np.sort(singularVals)
singularVals = singularVals[::-1]
Sigma = np.zeros((len(U), len(V)))
for i in range(min(rows, columns)):
    Sigma[i, i] = singularVals[i]
#Somehow these eigenvalues/eigenvectors are not working out exactly
#U, Sigma, and V.T should be the correct matrices returned. 
#np.matmul(np.matmul(U, Sigma), V.T)
#Somehow these are the exact same values as SVD but return different numbers
#Specifically, some of the orthonormal eigenvectors are the negative
#of what they should be? Lol why