"""
RPCA demo - lobbytest

ECE 510 

python version 3.7.3

Spring 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pcp import pcp
from matplotlib.animation import FuncAnimation

## Animation plot update
def updatefig(n):
    im1.set_array(np.reshape(X[:, n], (80, 64)).T)
    im2.set_array(np.reshape(L[:, n], (80, 64)).T)
    im3.set_array(np.reshape(S[:, n], (80, 64)).T)
    return im1, im2, im3

## Load data
data = loadmat("lobby.mat")
X = data['X']

L, S = pcp(X)

## Show the resulting videos
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
ax1.set_title("original video")
ax2.set_title("low-rank component")
ax3.set_title("sparse component")

im1 = ax1.imshow(np.reshape(X[:, 0], (80, 64)).T, cmap='gray')
im2 = ax2.imshow(np.reshape(L[:, 0], (80, 64)).T, cmap='gray')
im3 = ax3.imshow(np.reshape(S[:, 0], (80, 64)).T, cmap='gray')

ani = FuncAnimation(fig, updatefig, interval=50, frames=X.shape[1])
plt.show()