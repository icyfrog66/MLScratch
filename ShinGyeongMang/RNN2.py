# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:42:52 2018

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
#First, we have 31 by 1 input
#weights 0 must be 31 by hidden dimension, 
weightsXH = np.random.randn(vocab_size, innerDim)/np.sqrt(vocab_size)
#input1[:, 0, :] has shape 31 by 1
weightsHY = np.random.randn(innerDim, vocab_size)/np.sqrt(innerDim)
weightsHH = np.random.randn(innerDim, innerDim)/np.sqrt(innerDim)
HiddenState1 = np.matmul(input1[:, 0, :].T, weightsXH)
Output1 = np.matmul(HiddenState1, weightsHY)
#Output an m
predictions = Output1
predictions -= np.max(predictions)
predictions = np.e**predictions/(np.sum(np.e**predictions))
errors = predictions.copy()
errors[0, np.argmax(errors)] -= 1
dHY = 0
dHY += np.matmul(HiddenState1.T, errors)
dHidden1 = np.matmul(weightsHY, errors.T)

#Shape is currently 1 by 31, so it needs to be shifted
tanHidden1 = tanh(HiddenState1)
#Still need to add the second input in
HiddenState2 = np.matmul(HiddenState1, weightsHH) + np.matmul(input1[:, 1, :].T, weightsXH)
#And so on...
Output2 = np.matmul(HiddenState2, weightsHY)