# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:39:16 2018

@author: Anthony
"""
import nltk
import numpy as np
import re
text1 = open('Alice.txt', 'r', encoding = 'utf-8').read()
text1 = text1.lower()
text1 = re.sub("[^a-zA-Z\.!,']"," ", text1)
#text2 = open('SAOn.txt', 'r', encoding = 'utf-8').read()
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
sentences = sent_tokenize(text1)
words = word_tokenize(text1)
fDist = FreqDist(words)

unknown_token = 'UNK'
vocabSize = 8000
mostCommonWords = fDist.most_common(vocabSize - 1)
index_to_word = [x[0] for x in mostCommonWords]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

for i, sent in enumerate(sentences):
    sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in sentences])

trainingEx = X_train[10]
class RNNNumpy:
     #Assumes just one hidden layer
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))