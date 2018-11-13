# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:07:25 2018

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

class RNN:
    
    def __init__(self, vocab_size = 31, innerDim = 50, seqLength = 20):
        self.weightsXH = np.random.randn(vocab_size, innerDim)/np.sqrt(vocab_size)
        self.weightsHY = np.random.randn(innerDim, vocab_size)/np.sqrt(innerDim)
        self.weightsHH = np.random.randn(innerDim, innerDim)/np.sqrt(innerDim)
        self.vocab_size = vocab_size
        self.innerDim = innerDim
        self.seqLength = seqLength
        
    def forwardOnly(self, inputVect):
        #First print out the first hidden state here
        hiddenStates = [np.matmul(inputVect[:, 0, :].T, self.weightsXH)]
        yVals = [np.matmul(hiddenStates[0], self.weightsHY)]
        for i in range(1, self.seqLength):
            hiddenToHidden = np.matmul(tanh(hiddenStates[i-1]), self.weightsHH)
            newStepToHidden = np.matmul(inputVect[:, i, :].T, self.weightsXH)
            hiddenStates.append(hiddenToHidden + newStepToHidden)
            yVals.append(np.matmul(hiddenStates[i], self.weightsHY))
        return hiddenStates, yVals
    
    def forwardAndBackward(self, inputVect):
        return 0
rnn = RNN()
hid, yv = rnn.forwardOnly(input1)
            