# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:37:28 2018

@author: Anthony
"""

import numpy as np

#Input: Vector of size 2
inputSize = 2
Nodes1 = 4
learningRate = 0.01

#Use np.append on the input
#np.matmul for matrices
numPoints = 100
inputEntries = np.random.rand(numPoints, inputSize)

Input = np.ones((1, inputSize))
w1 = np.random.rand(inputSize + 1, Nodes1)
w2 = np.random.rand(Nodes1 + 1, 1)


epochs = 10000
for i in range(epochs):
    curEntry = inputEntries[i % len(inputEntries), :]
    curEntry = np.append(curEntry, 1)
    #curEntry = np.array(curEntry)
    firstLayerNodes = np.matmul(curEntry, w1)
    firstLayerNodes = np.tanh(firstLayerNodes)
    firstLayerNodes = np.append(firstLayerNodes, 1)
    output = np.matmul(firstLayerNodes, w2)
    #print(output[0])
    w2 -= w2 * learningRate
#Error is equal to the output value, assuming that the correct
#value is 0
    
#Wow that derivative took some time to calculate
    w1 -= w1 * learningRate * (np.matmul(curEntry, w1)) * \
        (1 - np.tanh(np.matmul(curEntry, w1) * np.matmul(curEntry, w1)))

#w1 weights go down much faster than w2 weights