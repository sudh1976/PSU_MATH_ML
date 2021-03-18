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
        L, C = op(Y, test='mnist', lam = lam, lam_s = lam_c)
        Yhat = L + C
        heatmap[l, lc] = norm(Y - Yhat, ord='fro') / norm(Y, ord='fro')

        print(l, lc)

plt.imshow(heatmap)
plt.colorbar()
plt.title("Heat Map")
plt.savefig('digitTest_heatmap_1')
plt.show()


# l2norm_C = []

# for ii in range(Y.shape[1]):
#     column_norm = norm(C[:, ii])
#     l2norm_C.append(column_norm)

# outliers = np.argwhere(np.array(l2norm_C) > 2)
# print(outliers)
# for outlier in np.squeeze(outliers):
#     Xtemp = Y[:, outlier].reshape(28, 28)
#     plt.imshow(Xtemp.T)
#     plt.show()
# classes = [1, 7]
# figs, axes = plt.subplots(3, 3, figsize=(12, 8))
# plt.set_cmap('gray')
# axes = axes.ravel()
# figInd = 0
# for ii in classes:
#     Xtemp = X[:, ii + ii * 200].reshape(28,28)
#     axes[figInd].imshow(Xtemp.T)
#     figInd+=1
#     Xtemp = X[:, ii + 50 + ii * 200].reshape(28,28)
#     axes[figInd].imshow(Xtemp.T)
#     figInd+=1
#     Xtemp = X[:, ii + 100 + ii * 200].reshape(28,28)
#     axes[figInd].imshow(Xtemp.T)
#     figInd+=1

# plt.suptitle('Example Images')
# plt.show()

# plt.stem(l2norm_C)
# plt.show()