"""
RPCA demo - lobbytest

ECE 510 

python version 3.7.3

Spring 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from op import op
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm

## Animation plot update
def updatefig(n):
    im1.set_array(np.reshape(X[:, n], (80, 64)).T)
    im2.set_array(np.reshape(L[:, n], (80, 64)).T)
    im3.set_array(np.reshape(S[:, n], (80, 64)).T)
    return im1, im2, im3

## Load data
data = loadmat("lobby.mat")
X = data['X']

# lam_list = np.linspace(0.001, 0.009, 10)
# print(lam_list)
# heatmap = np.zeros((len(lam_list), len(lam_list)))
# for l in range(len(lam_list)):
#     for ls in range(len(lam_list)):
#         lam = lam_list[l]
#         lam_s = lam_list[ls]
#         L, S, errL, errS = op(X, test='lobby', lam=lam, lam_s=lam_s)
#         Xhat = L + S
#         heatmap[l, ls] = norm(X - Xhat, ord='fro') / norm(X, ord='fro')
#
#         print(l, ls)
#
# plt.imshow(heatmap)
# plt.title("Heat Map")
# plt.show()

L, S = op(X, test='lobby', lam=0.008, lam_s=0.006)

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