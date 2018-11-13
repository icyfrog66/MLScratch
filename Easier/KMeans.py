# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:58:35 2018

@author: Anthony
"""

import numpy as np
import matplotlib.pyplot as plt
#Dimension is 2: can choose another value
numPoints = 1000
Data = np.zeros((numPoints, 2))
for i in range(numPoints):
    Data[i][0] = np.random.uniform(0, numPoints)
    Data[i][1] = np.random.uniform(0, numPoints)
    
plt.scatter(Data[:, 0], Data[:, 1])

numMeans = 5

means = np.random.rand(numMeans, 2)
listsDict = {}
for i in range(numMeans):
    listsDict[i] = []
iterations = 100

def closestMean(meanPoints, point):
    closestMean = 0
    smallestDistance = 10**20
    for i in range(len(meanPoints)):
        curDist = np.linalg.norm(point - meanPoints[i, :])
        if curDist < smallestDistance:
            smallestDistance = curDist
            closestMean = i
    return i

for i in range(iterations):
    for j in range(numPoints):
        curPoint = Data[j, :]
        listsDict[closestMean(means, curPoint)].append(curPoint)
    for j in range(len(means)):
        curList = listsDict[j]
        if len(curList) == 0:
            continue
        totalVec = np.array([0.0, 0.0])
        for val in curList:
            totalVec += val.astype(float)
        totalVec /= len(curList)
        means[j] = totalVec
    for j in range(len(means)):
        listsDict[i] = []
    print(i)