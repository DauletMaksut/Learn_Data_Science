import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cluster
from scipy.signal import convolve2d
image = plt.imread('Lenna.png')
# image1 = plt.imread ('phantom.png')
# weights of neuron
# h = np.array ([[1 , -1] ,[ -1 ,1]])
h = np . array ([[1 , 1 ,1 ,1 ,1] ,[ -1 , -1 , -1 , -1 , -1]])
# h = np . ones ((10 ,10))
y = convolve2d ( image , h , mode = 'same')
# plt . figure (1)
# plt.subplot(121)
# plt . imshow ( image , interpolation = 'nearest', cmap ='gray')
# plt.subplot(122)
# plt . imshow (y , interpolation = 'nearest' , cmap ='gray')
plt.figure(1)
plt.subplot(231)
plt.imshow ( image , interpolation = 'nearest', cmap ='gray')
plt.title("Original image")
x = 10
while x < 51:
    plt.subplot(230 + x//10 + 1)
    h = np.ones((x, x))
    y = convolve2d ( image , h , mode = 'same')
    plt.imshow (y , interpolation = 'nearest' , cmap ='gray')
    plt.title("ones of:" + str(x))
    x += 10

plt.show()







