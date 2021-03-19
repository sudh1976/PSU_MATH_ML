import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import norm
from op import op

"""
RPCA demo - digitTest
"""

## Load data
data = loadmat("mnistSubset.mat")

X = data['X']
trueLabels = data['trueLabels'][:,0]

Y = np.array([]).reshape(784, 0)

Xk = X[:, np.argwhere(trueLabels==2)[:,0]]
Y = np.append(Y, Xk[:, :200], axis=1)

Xk = X[:, np.argwhere(trueLabels==8)[:,0]]
Y = np.append(Y, Xk[:, 100:111], axis=1)

print(Y.shape)

lam_list = np.linspace(0.005, 0.009, 10)
print(lam_list)
heatmap = np.zeros((len(lam_list), len(lam_list)))
for l in range(len(lam_list)):
    for lc in range(len(lam_list)):
        lam = lam_list[l]
        lam_c = lam_list[lc]
        L, C, _ = op(Y, test='mnist', lam = lam, lam_s = lam_c)
        Yhat = L + C
        heatmap[l, lc] = norm(Y - Yhat, ord='fro') / norm(Y, ord='fro')

        print(l, lc)

plt.imshow(heatmap)
plt.colorbar()
plt.title("Heat Map")
plt.savefig('digitTest_heatmap_1')
plt.show()


# calculate l2 norm of each column
L, C, _ = op(Y, test='mnist', lam=0.01, lam_s=0.01)
l2norm_C = []
for ii in range(Y.shape[1]):
    column_norm = norm(C[:, ii])
    l2norm_C.append(column_norm)

for ii in range(Y.shape[1]):

    # outliers in ones
    if 0 <= ii < 200:
        if l2norm_C[ii] > 3:
            Xtemp = Y[:, ii].reshape(28, 28)
            plt.imshow(Xtemp.T)
            plt.show()

    # outliers in sevens
    else:
        if l2norm_C[ii] < 2:
            Xtemp = Y[:, ii].reshape(28, 28)
            plt.imshow(Xtemp.T)
            plt.show()

plt.title("Outlier Pursuit for 200 ones and 11 sevens")
plt.ylabel('L2 norm of Ci')
plt.xlabel('image index i')
plt.stem(l2norm_C)
plt.show()