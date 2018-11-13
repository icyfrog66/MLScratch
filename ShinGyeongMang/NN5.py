# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 13:49:17 2018

@author: Anthony
"""

#Lecture 6

import pandas as pd
import numpy as np

league = pd.read_csv('Data/games.csv')

x = league.iloc[:, [2, 5, 6, 7, 8, 9, 10, 26, 27, 28, 29, 30, 51, 52, 53, 54, 55]]
y = league.iloc[:, [4]]


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train.values.ravel())
y_pred = knn.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
#9829/10297, 95% accuracy with KNN

#Zero Mean, then Normalize
#This x is now different
#For images, subtract the mean image
x -= np.mean(x, axis = 0)
#Skip this step for images
x /= np.std(x, axis = 0)

#Dim 17
w1 = np.random.rand(17, 10)/np.sqrt(17)
b1 = np.zeros((1, 10))
w2 = np.random.rand(10, 2)/np.sqrt(10)
b2 = np.zeros((1, 2))
gamma1 = np.ones((1, 10))
beta1 = np.ones((1, 10))

def tanh(vector):
    return (np.e**vector - np.e**(-vector))/(np.e**vector + np.e**(-vector))
batchSize = 15
batchOne = np.array(x.iloc[0:batchSize, :])
yOne = np.array(y.iloc[0:batchSize])
x1 = np.matmul(batchOne, w1) + b1 #Batch size by 2nd weight dimension
mean1 = np.mean(x1, axis = 0)
var1 = np.var(x1, axis = 0)
x1Norm = (x1 - mean1)/(np.sqrt(var1) + 1e-8)
x1PreTan = np.multiply(x1Norm, gamma1) + beta1
tanhx1 = tanh(x1)
x2 = np.matmul(tanhx1, w2) + b2

x2Normal = x2 - np.max(x2, axis = 1).reshape((batchSize, 1))
predictions = np.e**(x2Normal)/np.sum(np.e**(x2Normal))
errors = predictions.copy()
for i in range(5):
    errors[i][yOne[i]] -= 1
dx2 = np.matmul(errors, w2.T)
dw2 = np.matmul(x1.T, errors) #Ends up the correct shape?
dxDownStream = np.multiply(dx2, 1 - dx2**2)
dGamma = np.sum(np.matmul(dxDownStream.T, x1Norm), axis = 0)
dBeta = np.sum(dxDownStream, axis = 0)
#Now batch norm layer
dxDownAfterGamma = np.multiply(dxDownStream, gamma1)
fPrime = 8/9 * x1 
gPrime = (2*(x1 - mean1))/(2*np.sqrt(var1 + 1e-8))
derivMult = (fPrime * np.sqrt(var1) - (x1 - mean1) * gPrime)/var1
dx1 = derivMult * dxDownAfterGamma #Np multiply, not matmul
dw1 = np.matmul(batchOne.T, dx1)
