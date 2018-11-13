# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:26:20 2018

@author: Anthony
"""

#No color

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def relu(val):
    new = np.zeros(val.shape)
    new[val < 0] = 0
    new[val > 0] = 1
    result = np.multiply(new, val)
    return result

def reluDeriv(val):
    val2 = val.copy()
    val2[val2 < 0] = 0
    val2[val2 > 0] = 1
    return val2

"""

def convolution(original, conv, numFilters = 5, stride = 1):
    conv = np.flip(conv, 0)
    conv = np.flip(conv, 1)
    final = np.zeros((original.shape[0] - conv.shape[0] + 1, 
                      original.shape[1] - conv.shape[0] + 1, 
                      numFilters))
    current = np.zeros((original.shape[0] - conv.shape[0] + 1, 
                      original.shape[1] - conv.shape[0] + 1))
    for k in range(0, numFilters):
        for i in range(0, original.shape[0] - conv.shape[0], stride):
            for j in range(0, original.shape[1] - conv.shape[1], stride):
                current[i][j] = sum(sum(sum(np.multiply(original[i:i+
                       conv.shape[0], j:j+conv.shape[1]], conv))))
        final[:, :, k] = current
        
    return final
"""
#Initialize ConvList on compile
#No Zero Padding
def convolution(original, conv, stride = 1):
    conv = np.flip(conv, 0)
    conv = np.flip(conv, 1)
    final = np.zeros((original.shape[0] - conv.shape[0] + 1, 
                      original.shape[1] - conv.shape[0] + 1))
    for i in range(0, int((original.shape[0] - conv.shape[0])/stride + 1)):
        for j in range(0, int((original.shape[1] - conv.shape[1]/stride) + 1)):
            final[i][j] = sum(sum(sum(np.multiply(original[i:i+
                   conv.shape[0], j:j+conv.shape[1]], conv))))
        
    return final

#Precondition: even dimension, although maybe not completely necessary,
#it is helpful. Original should also be 3d
#returns result and maxArgs
#maxArgs returned is weird
def pool(original, dim = 2, stride = 2):
    dim1 = int(original.shape[0]/2)
    dim2 = int(original.shape[1]/2)
    dim3 = original.shape[2]
    result = np.zeros((dim1, dim2, dim3))
    argDict = {}
    for i in range(0, dim1):
        for j in range(0, dim2):
            for k in range(dim3):
                curMatrix = original[i:i+dim, j:j+dim, k]
                maxVal = -10**20
                maxIndex = [0, 0]
                for l in range(dim):
                    for m in range(dim):
                        if curMatrix[l][m] > maxVal:
                            maxVal = curMatrix[l][m]
                            maxIndex = [l, m]
                result[i, j, k] = maxVal
                argDict[i, j, k] = maxIndex
    return result, argDict
#(a, b, c) returns to 2a, 2b, 2b + argDict[2a, 2b, 2c], elementwise



img = mpimg.imread('Images/mosaic.png')
#img = img[1:100, 1:100, 0]
imgplot = plt.imshow(img[100:250, 450:800, :])
img = img[100:250, 450:800, :]
conv1List = []
numLayer1 = 5
for i in range(numLayer1):
    conv1List.append(np.random.rand(3, 3, 4))
#Not same size: (img[0] - conv[0] )/stridesize + 1
final1 = np.zeros((148, 348, numLayer1))
for i in range(numLayer1):
    final1[:, :, i] = convolution(img, conv1List[i])
    
reluD1 = reluDeriv(final1)
relu1 = relu(final1)
result, argDict = pool(relu1)

conv2List = []
numLayer2 = 3
#Third dim of this must be same as num of conv in layer 1
for i in range(numLayer2):
    conv2List.append(np.random.rand(3, 3, 5))
final2 = np.zeros((72, 172, numLayer2))
for i in range(numLayer2):
    final2[:, :, i] = convolution(result, conv2List[i])

reluD2 = reluDeriv(final2)
relu2 = relu(final2)
result2, argDict2 = pool(relu2)

conv3List = []
numLayer3 = 1

#3 from previous layer
for i in range(numLayer3):
    conv3List.append(np.random.rand(3, 3, 3))
final3 = np.zeros((34, 84, numLayer3))
for i in range(numLayer3):
    final3[:, :, i] = convolution(result2, conv3List[i])

reluD3 = reluDeriv(final3)
relu3 = relu(final3)
result3, argDict3 = pool(relu3)

denseInput = np.ndarray.flatten(result3)
denseInput = denseInput.reshape(denseInput.shape[0], 1)
w1 = np.random.rand(714, 256)
b1 = np.random.rand(1, 256)

w2 = np.random.rand(256, 4)
b2 = np.random.rand(1, 4)


x1 = np.matmul(denseInput.T, w1) + b1
x1ReluDer = reluDeriv(x1)
x1Relu = relu(x1) 

x2 = np.matmul(x1, w2) + b2


x2Normal = (x2 - np.min(x2))/(np.max(x2) - np.min(x2))
x2Class = 0
predictions = np.e**x2Normal / np.sum(np.e**x2Normal)
errors = predictions.copy()
errors[[x2Class], 0] -= 1

dx2 = np.matmul(errors, w2.T)
dw2 = np.matmul(x1.T, errors)
dx2Down = np.multiply(dx2, x1ReluDer)

dx1 = np.matmul(dx2Down, w1.T)
dw1 = np.matmul(denseInput, dx2Down)
#No relu was done here
dx1Down = dx1

dx1Down = dx1Down.reshape(result3.shape)

def convertToPrePoolDeriv(downStream, poolDict):
    prePool = np.zeros((2 * downStream.shape[0], 2 *downStream.shape[1], 
                        downStream.shape[2]))
    for i in range(downStream.shape[0]):
        for j in range(downStream.shape[1]):
            for k in range(downStream.shape[2]):
                val = poolDict[i, j, k]
                prePool[2 * i + val[0], 2 * j + val[1], k] = 1
    return prePool
lastPrePool = convertToPrePoolDeriv(dx1Down, argDict3)

#36 86 3 for r2Deriv
r2DerivShape = result2.shape
#conv3List: [0] has shape (3, 3, 3), lol 
conv3Copy = conv3List[0].copy()
#r2Upstream = np.zeros(r2DerivShape)
#convDerivs = []
lr = 0.005
#for i in range(len(conv3List)):
#    convDerivs.append(np.zeros(conv3List[i].shape))
def convertToPreConvDeriv(lastPrePool, convList, resultShape, resultArr):
    rUpStream = np.zeros(resultShape)
    firstDim = convList[0].shape[0]
    secondDim = convList[0].shape[1]
    biasAdj = np.zeros(len(convList))
    #thirdDim = convList[0].shape[2]
    for i in range(lastPrePool.shape[0]):
        for j in range(lastPrePool.shape[1]):
            for l in range(len(convList)):
                convVal = resultArr[i:i+firstDim, j:j+secondDim, 
                                  :] * lastPrePool[i, j, 0]
                #print(convVal.shape)
                #print(convList[l].shape)
                convVal = reluDeriv(convVal)
                rUpStream[i:i+firstDim, j:j+secondDim, :] += convVal
                #convDerivs[l] -= convVal
                convList[l] -= lr * convVal
                biasAdj[l] += sum(sum(sum(convVal)))
    return rUpStream, convList, biasAdj
#This actually finishes the adjustments as well
r2UpStream, conv3List, biasAdj2 = convertToPreConvDeriv(lastPrePool, conv3List, r2DerivShape, result2)

secondPrePool = convertToPrePoolDeriv(r2UpStream, argDict2)
r1DerivShape = result.shape
conv2Copy = conv2List.copy()
r1UpStream, conv2List, biasAdj1 = convertToPreConvDeriv(secondPrePool, conv2List, r1DerivShape, result)
firstPrePool = convertToPrePoolDeriv(r1UpStream, argDict)
r0DerivShape = img.shape
conv1Copy = conv1List.copy()
r0UpStream, conv1List, biasAdj0 = convertToPreConvDeriv(firstPrePool, conv1List, r0DerivShape, img)
#np.multiply is element wise multiply




#np dstack

#Conv Layer requires number of filters, size, stride, padding
#Pooling layer: takes in size of pool and stride, usually F=2, S=2









#Conv Shape 40 by 40 by 4, for example




"""
Architecture:
1 Convolution, Tanh (probably)
1 Convolution, Tanh, Max Pool
2 Dense
Probably do this by hand to be able to do it
"""


