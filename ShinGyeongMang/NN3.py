# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:45:47 2018

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

print(tanh(vect))

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
dw1 = np.matmul(x1.T, errors)
dxDownStream = np.multiply(dx2, 1 - dx2**2)

dx1 = np.matmul(dxDownStream, w1.T)
dw0 = np.matmul(vect.T, dxDownStream)