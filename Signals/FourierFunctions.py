# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 20:56:46 2018

@author: Anthony
"""

from scipy.fftpack import fft
def fourier_extraction(this_it):
  this_it_features = []
  for i in range(len(this_it)):
    cur = this_it[i]
    curFFT = fft(cur)
    curABS = np.abs(curFFT)
    this_it_features.append(feature_extraction(curABS))  
    
    
    
#Improved Version
from scipy.fftpack import fft
def fourier_extraction(this_it):
  cur = fft(this_it)
  curABS = np.abs(curFFT)
  return feature_extraction(curABS)      


#Looks like some windowing is necessary   
  

#Trying an argmax situation
#Likely just need to keep taking small windows: stick with the standard
#normal window function for now
  

#Take in a 4000 by 10000 array
#For each in the 4000, get featuers from the next 10000