# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:36:10 2018

@author: Anthony
"""

import numpy as np
import matplotlib.pyplot as plt

numDims = 2

numPoints = 5000
x = 2 * np.random.rand(numPoints, 3) - 1
#x[:, 0] = 0.5
y = np.ones((numPoints, 1))
#3 parameters
for i in range(numPoints):
    if sum(x[i, :]) < 0:
        y[i, 0] = -1

params = 2 * np.random.rand(3, 1) - 1

def sigmoid(val):
    return 1/(1+np.e**-val)

def sigmoidDeriv(val):
    return val * (1  - val)

lr = 0.005

numEpochs = 300
for i in range(numEpochs):
    hx = np.matmul(x, params)
    predictions = sigmoid(hx)
    predictions2 = np.ones(predictions.shape)
    for j in range(len(predictions)):
        if predictions[j]  < 0.5:
            predictions2[j] = -1
    svError = 1 - np.multiply(predictions2, hx)
    for j in range(len(svError)):
        svError[j] = max(0, svError[j])
    derivChange = np.matmul(svError.T, x)
    params -= lr / numPoints * derivChange.T
    print('change')
    print(derivChange)
    print('params')
    print(params)
    #print(i)













"""
#This assumes 2 dimensions, although a constant could be used instead
numPoints = 200
data = np.random.rand(numPoints, 2)
classes = np.zeros((numPoints, 1))
for i in range(numPoints):
    if data[i, 0] + data[i, 1] > 1:
        classes[i, 0] = 1

x = data[:, 0]
y = data[:, 1]
plt.scatter(x, y)

weights = np.random.rand(1, 2)
#y = mx + b, m is weights[0, 0], b is weights[0, 1]
def distanceToPlane(point, weights):
    return np.abs(point[0] * weights[0][0] - point[1] + weights[0][1])/ \
        np.sqrt(weights[0][0]**2 + 1)
        
def closestPoints(points, weights, numPoints = 2):
    distances = np.zeros((len(points), 1))
    for i in range(len(points)):
        distances[i, 0] = distanceToPlane(points[i, :], weights)
    closest = []
    for i in range(numPoints):
        index = np.argmin(distances)
        thePoint = points[index, :]
        closest.append(thePoint)
        distances[index] = 10**20
    return closest

#Assume 2 support vectors
"""