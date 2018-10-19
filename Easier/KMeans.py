# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:58:35 2018

@author: Anthony
"""

import numpy as np
import matplotlib.pyplot as plt
constant = 1000
numPoints = constant
Data = np.zeros((numPoints, 3))
for i in range(numPoints):
    Data[i][0] = np.random.uniform(0, constant)
    Data[i][1] = np.random.uniform(0, constant)
    
plt.plot(Data[:, 0], Data[:, 1], 'ro')

numMeans = 5