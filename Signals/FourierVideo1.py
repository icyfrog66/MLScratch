# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:41:58 2018

@author: Anthony
"""

import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from scipy.fftpack import fft

rate, data = wav.read('mint.wav')
val = 24000
interesting = data[:, 0]
interesting = interesting[20000: int(20000+val)]
ffResult = fft(interesting)
ffABS = np.abs(ffResult)
time = np.linspace(0., rate, num = len(interesting))
plt.plot(time, ffABS, 'ro')
print(np.argmax(ffABS))
#plt.plot(time, interesting, 'ro')

pnts = len(interesting)
#Creates the points in time? Normalizes
fourTime = np.array(range(0,pnts))/pnts
fCoefs   = np.zeros((len(interesting)),dtype=complex)

#WHen Plotted, this is just a circle
for i in range(pnts):
    exponential = np.e**(-1j*2*np.pi*i*fourTime)
#dotResult = np.multiply(exponential, interesting)
#Normalize the dot sum, seems like
    fCoefs[i] = np.abs(np.dot(exponential, interesting)/pnts)
    
ampls = 2*np.abs(fCoefs)

# compute frequencies vector
hz = np.linspace(0,int(rate/2),num=int(np.floor(pnts/2.)+1))

#plt.stem(hz,ampls[range(0,len(hz))])
#plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude (a.u.)')
#plt.xlim(0,4000)
#plt.show()

"""from scipy.fftpack import fft
curFFT = np.abs(fft(Y0to8_QTP29_raw_windows[8000:12000][0, 1000:1100]))
print(np.max(curFFT))
print(np.argmax(curFFT))
time = np.linspace(0., len(curFFT), num = len(curFFT))
import matplotlib.pyplot as plt
plt.plot(time, curFFT, 'ro')"""


"""from scipy.fftpack import fft
stored = Y0to8_QTP29_raw_windows[32000:36000][0, 100:200]
stored = stored - np.mean(stored)
curFFT = np.abs(fft(stored))
print(np.max(curFFT))
print(np.argmax(curFFT[len(curFFT)/100:len(curFFT)*2/5]))
print(np.max(curFFT[len(curFFT)/100::len(curFFT)*2/5]))
time = np.linspace(0., len(curFFT), num = len(curFFT))
import matplotlib.pyplot as plt
plt.plot(time, curFFT, 'ro')"""


def feature_extraction2(this_it, stride = 100):
  this_it_features = []
  for i in range(len(this_it)):
      f_list = []
      for j in range(len(this_it[i, :])/stride - 1):
          stored = this_it[i, stride*j:stride*(j+1)]
          stored -= np.mean(stored)
          curFFT = np.abs(fft(stored))
          f_list.append(np.argmax(curFFT[len(curFFT)/stride:len(curFFT)*9/20]))
      this_it_features.append(f_list)
  return this_it_features
Y0_features2 = feature_extraction2(Y0to8_QTP29_raw_windows[:4000])
#First Strategy: no denoising, just standard fourier

#Get the top every 100: hard to get the 2nd most anyways