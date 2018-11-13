# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:42:32 2018

@author: Anthony
"""

import nltk
import numpy as np
import re
text1 = open('Alice.txt', 'r', encoding = 'utf-8').read()
text1 = text1.lower()
text1 = re.sub("[^a-zA-Z\.!,']"," ", text1)

chars = sorted(list(set(text1)))
char_to_int = dict((character, index) for index, character in enumerate(chars))
int_to_char = dict((index, character) for character, index in char_to_int.items())
vocab_size = len(chars)
seqLength = 20
input1 = np.zeros((vocab_size, seqLength, 1))
output1 = np.zeros((vocab_size, seqLength, 1))
for i in range(seqLength):
    input1[char_to_int[text1[i]], i, 0] = 1
    output1[char_to_int[text1[i + 1]], i, 0] = 1

def tanh(x):
    return (np.e**x + np.e**-x)/(np.e**x + np.e**-x)


innerDim = 50
prevH = np.random.randn(1, innerDim) #1 by 50
prevC = np.random.randn(1, innerDim) #1 by 50
dataPoint = input1[:, 0, :].T #1 by 31
WXG = np.random.randn(vocab_size, innerDim)
WXI = np.random.randn(vocab_size, innerDim)
WXF = np.random.randn(vocab_size, innerDim)
WXO = np.random.randn(vocab_size, innerDim)
WHG = np.random.randn(innerDim, innerDim)
WHI = np.random.randn(innerDim, innerDim)
WHF = np.random.randn(innerDim, innerDim)
WHO = np.random.randn(innerDim, innerDim)
yWeights = np.random.randn(innerDim, vocab_size)

def sigmoid(x):
    return 1/(1 + np.e**-x)

#Ignoring biases
cacheI = np.matmul(prevH, WHI) + np.matmul(dataPoint, WXI)
i1 = sigmoid(cacheI)#datapoint is a vector?
cacheF = np.matmul(prevH, WHF) + np.matmul(dataPoint, WXF)
f1 = sigmoid(cacheF)
cacheO = np.matmul(prevH, WHO) + np.matmul(dataPoint, WXO)
o1 = sigmoid(cacheO)
cacheG = np.matmul(prevH, WHG) + np.matmul(dataPoint, WXG)
g1 = tanh(cacheG)
cT = prevC*f1 + i1*g1
tanCTStore = tanh(cT)
hT = o1 * tanCTStore
#Seems like you would get a y value out of the hT
yOutput = np.matmul(hT, yWeights) #1 by 31
#Prediction of the next one is 5
preds1 = yOutput.copy()
preds1 -= np.max(preds1)
preds1 = np.e**preds1/(np.sum(np.e**preds1))
preds1[0, char_to_int[text1[1]]] -= 1
dYOut = np.matmul(preds1.T, hT)
dHT = np.matmul(preds1, yWeights.T)
do1 = dHT*tanCTStore
dtanCT = dHT * o1
dCT = (1 - cT**2)*dtanCT
dF1 = dCT*prevC
dprevC = dCT*f1
di1 = dCT*g1
dg1 = dCT*i1
#Could add bias derivatives here
dCacheI = di1*(1-cacheI)*cacheI
#Not sure if shapes are correct here
dprevH = 0
dprevH += np.matmul(dCacheI, WHI.T)
dWHI = np.matmul(prevH.T, dCacheI)
dWXI = np.matmul(dataPoint.T, dCacheI)

dCacheF = dF1*(1-cacheF)*cacheF
dprevH += np.matmul(dCacheF, WHF.T)
dWHF = np.matmul(prevH.T, dCacheF)
dWXF = np.matmul(dataPoint.T, dCacheF)

dCacheO = do1*(1-cacheO)*cacheO
dprevH += np.matmul(dCacheO, WHO.T)
dWHO = np.matmul(prevH.T, dCacheO)
dWXO = np.matmul(dataPoint.T, dCacheO)

dCacheG = di1*(1-cacheG**2)
dprevH += np.matmul(dCacheG, WHG.T)
dWHG = np.matmul(prevH.T, dCacheG)
dWXG = np.matmul(dataPoint.T, dCacheG)

