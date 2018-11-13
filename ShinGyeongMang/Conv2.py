# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 22:51:17 2018

@author: Anthony
"""

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

def convertToPrePoolDeriv(downStream, poolDict):
    prePool = np.zeros((2 * downStream.shape[0], 2 *downStream.shape[1], 
                        downStream.shape[2]))
    for i in range(downStream.shape[0]):
        for j in range(downStream.shape[1]):
            for k in range(downStream.shape[2]):
                val = poolDict[i, j, k]
                prePool[2 * i + val[0], 2 * j + val[1], k] = 1
    return prePool

img = mpimg.imread('Images/mosaic.png')
#img = img[1:100, 1:100, 0]
imgplot = plt.imshow(img[100:250, 450:800, :])
img = img[100:250, 450:800, :]
lr = 0.005

#This also assumes 3D, maybe doesn't?
class convNet():
    convWeights = []
    convBiases = []
    denseWeights = []
    denseBiases = []
    convDimensions = []
    poolDimensions = []
    denseDimensions = []
    #poolDicts = []
    #Maybe add poolDims but that might be confusing
    inputShape = None
    denseShape = None
    lr = 0.005
    inputs = None
    classes = None
    def __init__(self, learningRate = None, inShape = None):
        if learningRate != None:
            self.lr = learningRate
        if inShape != None:
            self.inputShape = (150, 350, 4)
            
    def setImageDim(self, dims):
        self.inputShape = dims
        #self.dimensions[0] = dims

    #Perhaps conv will automatically come with pooling
    #ConvDim should be a list
    #InputShape must be nonzero
    def convLayer(self, convDim, numMaps, stride = 1):
        self.convWeights.append([])
        self.convBiases.append([])
        if len(self.convDimensions) == 0:
            self.convDimensions.append([self.inputShape[0] - convDim[0] + 1, 
                                        self.inputShape[1] - convDim[1] + 1, 
                                        numMaps])
        else:
            self.convDimensions.append([self.poolDimensions[-1][0] - convDim[0] + 1,
                                        self.poolDimensions[-1][1] - convDim[1] + 1,
                                        numMaps])
        self.poolDimensions.append([int(self.convDimensions[-1][0]/2), 
            int(self.convDimensions[-1][1]/2), self.convDimensions[-1][2]])   
                #[int((self.convDimensions[-1][0] - convDim[0])/stride + 1),  
                 # int((self.convDimensions[-1][1] - convDim[1])/stride + 1), numMaps])
        for i in range(numMaps):
            curWeights = np.zeros(convDim)
            curWeights += np.random.ranf(curWeights.shape)
            self.convWeights[-1].append(curWeights)
            #Appending one number
            self.convBiases[-1].append(np.random.rand(1)[0])
        self.convBiases[-1] = np.array(self.convBiases[-1])
        self.convBiases[-1] = self.convBiases[-1].reshape(len(self.convBiases[-1]), 1)
    def poolLayer(self, size = 2, stride = 2):
        self.poolDimensions.append(np.array(self.convDimensions[-1])/2).astype(int)
    
    def comp(self):
        #for i in range(len(self.convDimensions)):
            #self.poolDimensions.append([int(self.convDimensions[i][0]/2), 
                    #int(self.convDimensions[i][1]/2), self.convDimensions[i][2]])
        return 0
    
    def denseLayer(self, outputSize):        
        if len(self.denseDimensions) == 0:
            temp = self.poolDimensions[-1]
            self.denseDimensions.append(temp[0] * temp[1] * temp[2])
        self.denseDimensions.append(outputSize)
        self.denseWeights.append(np.random.rand(
                self.denseDimensions[len(self.denseDimensions) - 2], \
                self.denseDimensions[len(self.denseDimensions) - 1]))
        self.denseBiases.append(np.random.rand(1, 
                    self.denseDimensions[len(self.denseDimensions) - 1]))
        
    def forwardAndBackward(self, img):
        upStream = img
        convOutputs = []
        convOutputsPostRelu = []
        reluDerivs = []
        poolOutputs = []
        poolDicts = []
        for i in range(len(self.convWeights)):
            curOutput = np.zeros((self.convDimensions[i]))
            curOutputPostRelu = np.zeros((self.convDimensions[i]))
            for j in range(len(self.convWeights[i])):
                curOutput[:, :, j] = convolution(upStream, 
                                                   self.convWeights[i][j])
            curReluDeriv = reluDeriv(curOutput)
            curOutputPostRelu = relu(curOutput)
            poolResult, poolDict = pool(curOutputPostRelu)
            reluDerivs.append(curReluDeriv)
            convOutputs.append(curOutput)
            convOutputsPostRelu.append(curOutputPostRelu)
            poolOutputs.append(poolResult)
            poolDicts.append(poolDict)
            upStream = poolResult
            #print(upStream.shape)   
        #Store loads of derivative shit here
        #Really only need to store weights of output in each layer: lol
        denseOutputs = []
        upStream = upStream.flatten()
        upStream = upStream.reshape(1, len(upStream))
        denseOutputs.append(np.copy(upStream))
        for i in range(len(self.denseWeights)):
            upStream = np.matmul(upStream, self.denseWeights[i]) + \
                                 + self.denseBiases[i]
            upStream = relu(upStream)
            denseOutputs.append(upStream)
            print(upStream.shape)
        predictions = np.copy(upStream)
        predNorm = (predictions - np.min(predictions))/  \
            (np.max(predictions) - np.min(predictions))
        probs = np.e**(predNorm)/np.sum(np.e**(predNorm))
        errors = np.copy(probs)
        dxDownStream = np.copy(errors)
        #print(dxDownStream.shape)
        #print(denseOutputs)
        for k in range(len(self.denseWeights) - 1, -1, -1):
            #print('1')
            dx = np.matmul(dxDownStream, self.denseWeights[k].T)
            dw = np.matmul(denseOutputs[k].T, dxDownStream)
            db = dxDownStream
            self.denseWeights[k] -= self.lr * dw
            self.denseBiases[k] -= self.lr*db
            #print('2')
            dxDownStream = np.multiply(dx, 1 - dx**2)
            #print(dxDownStream.shape)
        dxDownStream = dxDownStream.reshape(self.poolDimensions[-1])
        #Now at (17, 42, 1) shape
        lastPrePool = None
        for i in range(len(self.convWeights) - 1, -1, -1):
            lastPrePool = convertToPrePoolDeriv(dxDownStream, poolDicts[i]) 
            if i == 0:
                derivShape = img.shape
                finalInput = img
            else:
                derivShape = self.poolDimensions[i - 1]
                #print('noerror')
                finalInput = poolOutputs[i - 1]
                #print('error')
            #print('1')
            #print(lastPrePool.shape)
            #print(np.array(self.convWeights[i]).shape)
            #print(derivShape)
            #print(np.array(convOutputs[i]).shape)
            #print(np.array(poolOutputs[i].shape))
            dxDownStream, tempList, biasAdj = convertToPreConvDeriv(lastPrePool, 
                                self.convWeights[i], derivShape, finalInput)
            #print('2')
            #print('1')
            self.convWeights[i] = tempList
            #print('2')
            #print(biasAdj.shape)
            #print(np.array(self.convBiases[i]).shape)
            self.convBiases[i] -= lr * biasAdj.reshape(self.convBiases[i].shape)
            #print('3')
        #firstPrePool = convertToPrePoolDeriv(dxDownStream, poolDicts[0])
        #r0DerivShape = img.shape
        #r0UpStream, tempList, biasAdj0 = convertToPreConvDeriv(firstPrePool, conv1List, r0DerivShape, img)
        #self.denseBiases[0] -= lr*biasAdj0
        #self.convWeights[0] 
            
lol = convNet()
lol.setImageDim(img.shape)
#Implicitly assuming a relu layer in here
lol.convLayer([3, 3, 4], 5)
#Might not even need pooling, as pooling seems to be assumed automatically?
#It would be more annoying to determine if a conv layer had pooling or not lol
#lol.poolLayer()
lol.convLayer([3, 3, 5], 3)
lol.convLayer([3, 3, 3], 1)
lol.denseLayer(256)
lol.denseLayer(4)
lol.forwardAndBackward(img)
#dense dimensions: 714, 256, 4
#Still need to compile at the end


#lol.comp()
#lol.poolLayer()
#Nothing crashed so far lol
#lol.forwardAndBackward(img)
#Finally got integers lol

#Implicit Assumption: Only having one pool after 1 conv, not weird orders
#lol.convDimensions
#convolution(img, lol.convWeights[0][0])




#Center at zero: but this only helps the first layer
#If start at 0, all neurons will output the same thing. 
#If initialize normal distribution, std dev starts to shrink. 
#Initialize by scaling by number of inputs: rand(in, out)/np.sqrt(in)
#Each batch: [x - E(X)]/var(x), right before activation function, then scale/shift