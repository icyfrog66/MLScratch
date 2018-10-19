# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:11:05 2018

@author: Anthony
"""

import numpy as np
import matplotlib.pyplot as plt
numPoints = 100
Data = np.zeros((numPoints, 2))
slopeConstant = 2
bias = 3
for i in range(numPoints):
    Data[i][0] = np.random.uniform(-100, 100)
    Data[i][1] = Data[i][0] * slopeConstant + bias + np.random.uniform(-1, 1)

initialSlope = 1
initialBias = 1
learningRate = 0.0001
plt.plot(Data[:, 0], Data[:, 1], "ro")

numSessions = 100000
for i in range(numSessions):
    index = np.random.randint(0, numPoints)
    x = Data[index][0]
    y = Data[index][1]
    temp0 = initialSlope - (2*x*(initialSlope * x + initialBias - y)) * learningRate
    temp1 = initialBias - (2 * (initialSlope * x + initialBias - y)) * learningRate
    initialSlope = temp0
    initialBias = temp1
    
finalSlope = initialSlope
finalBias = initialBias
print(finalSlope)
print(finalBias)