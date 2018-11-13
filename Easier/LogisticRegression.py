# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:57:47 2018

@author: Anthony
"""

import numpy as np
import matplotlib.pyplot as plt

numDims = 2

numPoints = 50000
x = np.random.rand(numPoints, 3)
x[:, 0] = 1
y = np.zeros((numPoints, 1))
#3 parameters
for i in range(numPoints):
    if sum(x[i, :]) > 2:
        y[i, 0] = 1

params = np.random.rand(3, 1)

def sigmoid(val):
    return 1/(1+np.e**-val)

def sigmoidDeriv(val):
    return val * (1  - val)

lr = 0.05

numEpochs = 1000
for i in range(numEpochs):
    hx = np.matmul(x, params)
    predictions = sigmoid(hx)
    storage = (predictions - y)/(np.multiply(predictions, 1 - predictions))
    derivWeights = lr / numPoints * np.matmul(x.T, storage)
    #print(derivWeights)
    params -= derivWeights
    
    