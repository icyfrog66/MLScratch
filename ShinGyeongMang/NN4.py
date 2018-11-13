# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 13:23:27 2018

@author: Anthony
"""

import numpy as np

#Note: this fails if the number of samplse is too small and then
#there ends up being at least one class that is not picked at all
numSamples = 100
inputSize = 3
inputs = np.random.rand(numSamples, inputSize)
numClasses = 4
classes = np.random.randint(numClasses, size = numSamples)

def tanh(vector):
    return (np.e**vector - np.e**(-vector))/(np.e**vector + np.e**(-vector))

class net:
    weights = []
    biases = []
    dims = [5]
    inputDim = 5
    lr = 0.005
    inputs = None
    classes = None
    def __init__(self, learningRate = None):
        if learningRate != None:
            self.lr = learningRate
            
    def setInputSize(self, inputSize):
        self.inputDim = inputSize
        self.dims[0] = inputSize
        
    def add(self, outputSize):
        self.dims.append(outputSize)
        self.weights.append(np.random.rand(self.dims[len(self.dims) - 2], \
                                           self.dims[len(self.dims) - 1]))
        self.biases.append(np.random.rand(1, self.dims[len(self.dims) - 1]))
        
    def comp(self, inputs, classes):
        self.inputs = inputs
        self.classes = classes
        self.dims.append(len(set(classes)))
        self.weights.append(np.random.rand(self.dims[len(self.dims) - 2], \
                                           self.dims[len(self.dims) - 1]))
        self.biases.append(np.random.rand(1, self.dims[len(self.dims) - 1]))
    
    def train(self, epochs = 100):
        for i in range(epochs):
            for j in range(len(self.inputs)):
                curInput = self.inputs[j, :]
                curInput = curInput.reshape(1, curInput.shape[0])
                x_n = [curInput]
                curX = np.copy(curInput)
                predictions = None
                for k in range(len(self.weights)):
                    xOut = np.matmul(curX, self.weights[k]) + self.biases[k]
                    xOuttanh = tanh(xOut)
                    curX = xOuttanh
                    x_n.append(curX)
                    if k == len(self.weights) - 1:
                        predictions = xOut
                predNorm = (predictions - np.min(predictions))/  \
                            (np.max(predictions) - np.min(predictions))
                probs = np.e**(predNorm)/np.sum(np.e**(predNorm))
                errors = np.copy(probs)
                errors[0, self.classes[j]] -= 1
                dxDownStream = np.copy(errors)
                for k in range(len(self.weights) - 1, -1, -1):
                    dx = np.matmul(dxDownStream, self.weights[k].T)
                    dw = np.matmul(x_n[k].T, dxDownStream)
                    db = dxDownStream
                    self.weights[k] -= self.lr * dw
                    self.biases[k] -= self.lr*db
                    dxDownStream = np.multiply(dx, 1 - dx**2)
        
        
    def predict(self, vect):
        curX = vect
        for i in range(len(self.weights)):
            curX = np.matmul(curX, self.weights[i])
        return curX
    
    def predictAll(self):
        correct = 0
        total = len(self.inputs)
        for j in range(len(self.inputs)):
            curInput = self.inputs[j, :]
            curInput = curInput.reshape(1, curInput.shape[0])
            for i in range(len(self.weights)):
                curInput = np.matmul(curInput, self.weights[i])
            if np.argmax(curInput) == self.classes[j]:
                correct += 1
        print(correct/total)

theNet = net()
theNet.setInputSize(inputSize)
theNet.add(4)
theNet.add(5)
theNet.comp(inputs, classes)
theNet.train()
theNet.predictAll()