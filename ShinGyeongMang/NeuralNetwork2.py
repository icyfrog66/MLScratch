# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:40:52 2018

@author: Anthony
"""

import numpy as np

#Note: it appears that not much is learned, although maybe it is impossible
#because the network is completely random

numSamples = 30
inputs = np.random.rand(numSamples, 5)
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
    
    def train(self, epochs = 5):
        for epoch in range(epochs):
            for i in range(len(self.inputs)):
                currentEntry = inputs[i, :]
                nodes = []
                preActivations = []
                for j in range(len(self.weights)):
                    curPreAct = np.matmul(currentEntry, self.weights[j].T)
                    curPreAct = curPreAct + self.biases[j]
                    curNode = tanh(curPreAct)
                    preActivations.append(curPreAct)
                    nodes.append(curNode)
                finalValues = np.array(preActivations[len(preActivations) - 1])
                finalValues /= np.sum(finalValues)
                predictions = (np.e**finalValues)/(np.sum(np.e**finalValues))
                #print(predictions.shape)
                #print(predictions)
                losses = np.array(predictions)
                losses[0, self.classes[i]] -= 1
                curLosses = np.transpose(losses)
                for j in range(len(self.weights) - 1, 0, -1):
                    dWeights = np.matmul(np.transpose(nodes[j]), 
                                         np.transpose(curLosses))
                    dXPre = np.matmul(self.weights[j].T, curLosses)
                    dXPost = np.multiply(dXPre, 1- dXPre**2)
                    self.weights[j] -= self.lr * dWeights
                    curLosses = dXPost

    def predict(self, vect):
        curX = vect
        for i in range(len(self.weights)):
            curX = np.matmul(curX, self.weights[i])
        return curX
        #print(curX)

theNet = net()
theNet.setInputSize(5)
theNet.add(5)
theNet.add(5)
theNet.comp(inputs, classes)
theNet.train()
total = len(inputs)
correct = 0
for i in range(len(inputs)):
    #print(theNet.predict(theNet.inputs[i, :]))
    if np.argmax(theNet.predict(theNet.inputs[i, :])) == theNet.classes[i]:
        correct += 1
        #print(i)
print("correct: " + str(correct) + "out of total: " + str(total))
