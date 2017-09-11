import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from numpy import fft

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

f = open('ekg_data_2_offset.txt','r')

data = f.read()
raw = data.splitlines()
y = preprocessing.scale(raw)

#Smooth out the data...
y = runningMeanFast(y, 10)

sz = len(y)

Fs = 720.0  # sampling rate
Ts = sz/Fs # 3.0/Fs; # sampling interval
t = np.arange(0, Ts, 1.0/Fs) # time vector

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range

Y = fft.fft(y) # fft computing and normalization
Y = Y[range(int(n/2))]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq[:600], abs(Y[:600]),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

plt.show()

