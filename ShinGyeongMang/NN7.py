# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:31:38 2018

@author: Anthony
"""

import numpy as np

vect = np.array([1, 2, 3]).reshape(1, 3)
w1 = np.random.rand(3, 4)
b1 = np.random.rand(1, 4)
w2 = np.random.rand(4, 5)
b2 = np.random.rand(1, 5)

def tanh(vector):
    return (np.e**vector - np.e**(-vector))/(np.e**vector + np.e**(-vector))

lr = 0.001
rho1 = 0.9
rho2 = 0.9
firstW1Moment = 0
secondW1Moment = 0
firstW1Unbias = 0
secondW1Unbias = 0
firstW2Moment = 0
secondW2Moment = 0
firstW2Unbias = 0
secondW2Unbias = 0
t = 0 #Note that t should iteratively range up


x1 = np.matmul(vect, w1) + b1
tanhx1 = tanh(x1)
x2 = np.matmul(tanhx1, w2) + b2
tanhx2 = tanh(x2)
correctClasses = np.array([0])
results = tanhx2

x2Normal = (x2 - np.min(x2))/(np.max(x2) - np.min(x2))
predictions = np.e**(x2Normal)/np.sum(np.e**(x2Normal))
errors = np.copy(predictions)
errors[correctClasses[0], 0] -= 1

dx2 = np.matmul(errors, w2.T)
dw2 = np.matmul(x1.T, errors)
firstW1Moment = rho1 * firstW1Moment + (1 - rho1) * dw2
secondW1Moment = rho2 * secondW1Moment + (1 - rho1) * dw2 * dw2
firstW1Unbias = firstW1Moment / (1 - rho1**t)
secondW1Unbias = secondW1Moment / (1 - rho2**t)
w2 -= lr * firstW1Unbias / (secondW1Unbias + 1e-8)
dxDownStream = np.multiply(dx2, 1 - dx2**2)

dx1 = np.matmul(dxDownStream, w1.T)
dw1 = np.matmul(vect.T, dxDownStream)
#Not sure if rho should be reused here lmao, probably not a big deal
firstW2Moment = rho1 * firstW2Moment + (1 - rho1) * dw1
secondW2Moment = rho2 * secondW2Moment + (1 - rho1) * dw1 * dw1
firstW2Unbias = firstW2Moment / (1 - rho1**t)
secondW2Unbias = secondW2Moment / (1 - rho2**t)
w1 -= lr * firstW2Unbias / (secondW2Unbias + 1e-8)