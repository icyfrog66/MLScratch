# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:58:44 2018

@author: Anthony
"""

import numpy as np
import matplotlib.pyplot as plt
constant = 1000
numPoints = constant
Data = np.zeros((numPoints, 3))
for i in range(numPoints):
    Data[i][0] = np.random.uniform(0, constant)
    Data[i][1] = np.random.uniform(0, constant)
    
plt.plot(Data[:, 0], Data[:, 1], 'ro')
#Sorting automatically, maybe not helpful?
#Data = Data[np.argsort(Data[:, 0])]
"""minimumHalfIndex = 0
for i in range(1, numPoints):
    if Data[i-1][0] < constant/2 and Data[i][0] >= constant/2:
        minimumHalfIndex = i
        break
for i in range(int(np.floor(0.8 * numPoints))):
    if i >= minimumHalfIndex:
        if Data[i][1] >= constant/2:
            Data[i][2] = 3
        else:
            Data[i][2] = 2
    else:
        if Data[i][1] >= constant/2:
            Data[i][2] = 1
        else:
            Data[i][2] = 0"""
for i in range(int(np.floor(0.8 * numPoints))):
    if Data[i][0] >= constant/2:
        if Data[i][1] >= constant/2:
            Data[i][2] = 3
        else:
            Data[i][2] = 2
    else:
        if Data[i][1] >= constant/2:
            Data[i][2] = 1
        else:
            Data[i][2] = 0
numNeighbors = 3        
for i in range(int(np.ceil(0.8 * numPoints)), numPoints):
    distances = np.zeros((int(np.ceil(0.8 * numPoints)), 2))
    for k in range(int(np.ceil(0.8 * numPoints))):
        distances[k][0] = np.linalg.norm(Data[i, 0:2] - Data[k, 0:2])
        distances[k][1] = Data[k, 2]
    distances = distances[np.argsort(distances[:, 0])]
    #print(distances[0:numNeighbors, :])
    closestDistances = distances[0:numNeighbors, 1]
    #print(closestDistances)
    Data[i, 2] = np.argmax(np.bincount(closestDistances.astype(int)))
