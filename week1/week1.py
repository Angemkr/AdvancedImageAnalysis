## Week 1: live code from class


import numpy as np
import matplotlib.pyplot as plt

s = 3.123
x = np.arange(-np.ceil(s*4), np.ceil((s*4)+1)).reshape(-1,1)

g = np.exp(-x**2/(2*s**2))
g /= g.sum()    

fig, ax = plt.subplots()
ax.plot(x, g)
plt.show()

##
g = 1/(s*np.sqrt(2*np.pi)) * np.exp(-x**2/(2*s**2))
g.sum()

import scipy.ndimage
import skimage.io

filename = 'C:\\Users\\angel\\Desktop\\DTU\\3rd Semester\\Advanced Image Analysis\\week1\\week1_data\\fibres_xcth.png'
im =skimage.io.imread(filename)
im_g = scipy.ndimage.convolve(im,g)

# %% 
fig, ax = plt.subplots(1,2)
ax[0].imshow(im_g)
ax[1].imshow(im)
plt.show()
