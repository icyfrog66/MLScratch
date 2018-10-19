# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:58:15 2018

@author: Anthony
"""

import numpy as np
constant = 5
data = np.zeros((constant, constant))
#Randomly initialize the data matrix
for i in range(constant):
    for j in range(constant):
        data[i][j] = np.random.randint(2*constant*constant) 
#columns are the variables in this case
#dataNorm = np.zeros((constant, constant))

#for i in range(constant):
#   mean = sum(data[:, i])/constant
#   dataNorm[:, i] = (data[:, i] - mean)/(mean * constant)
        
cov = np.zeros((constant, constant))
#Calculates the covariance
for i in range(constant):
    for j in range(constant):
        if i > j:
            cov[i][j] = cov[j][i]
        else:
            """rowMean = sum(data[:, i])/constant
            colMean = sum(data[:, j])/constant
            rowVector = data[:, i] - rowMean
            colVector = data[:, j] - colMean
            cov[i][j] = (np.dot(rowVector, colVector))/(constant - 1)"""
            cov[i][j] = (np.dot((data[:, i] - sum(data[:, i])/constant), 
               (data[:, j] - sum(data[:, j])/constant)))/(constant - 1)

w, v = np.linalg.eig(cov)
eigIndex = np.argsort(w)
v = v[eigIndex]
pcaResult = np.dot(data, v)