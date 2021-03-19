"""
RPCA demo - syntheticTest
"""

import numpy as np
from numpy.random import randn, rand
from numpy.linalg import norm, svd
from op import op
import matplotlib.pyplot as plt

## Performance vs. rank and outliers number

N = 400
D = 400

def generate_matrix(rank, gam):
    D = 400
    N = 400
    r = rank
    gamma = gam

    A = randn(D, r)
    B = randn(int(N - gamma * N), r)
    C = randn(D, int(gamma * N))

    Ltrue = np.hstack((A @ B.T, np.zeros((D, int(gamma * N)))))  # low - rank matrix
    Strue = np.hstack((np.zeros((D, int(N - gamma * N))), C))  # sparse matrix
    Y = Ltrue + Strue

    return Y, Ltrue, Strue


zL = np.zeros((6, 5))
zC = np.zeros((6, 5))
zM = np.zeros((6, 5))
ranks = np.linspace(5, 35, 6)
gammas = np.linspace(0.05, 0.3, 5)
for ii, r in enumerate(ranks):
    for jj, gam_ in enumerate(gammas):
        print(r)
        print(gam_)
        # generate different synthetic set
        M, Ltrue, Strue = generate_matrix(int(r), gam_)
        # divide M into L and S
        L, S, _ = op(M, test='')
        errL = norm(Ltrue - L, ord='fro') / norm(Ltrue, ord='fro')
        errS = norm(Strue - S, ord='fro') / norm(Strue, ord='fro')
        Mest = L+S
        errM = norm(Mest - M, ord='fro') / norm(M, ord='fro')
        zL[ii, jj] = errL
        zC[ii, jj] = errS
        zM[ii, jj] = errM


plt.imshow(zL, extent=[0.05*N, 0.3*N, 5, 35], aspect='auto')
plt.title("matrix L error")
plt.xlabel("Outlier numbers")
plt.ylabel("Rank")
plt.colorbar()
plt.show()

plt.imshow(zC, extent=[0.05*N, 0.3*N, 5, 35], aspect='auto')
plt.title("matrix C error")
plt.xlabel("Outlier numbers")
plt.ylabel("Rank")
plt.colorbar()
plt.show()

plt.imshow(zM, extent=[0.05*N, 0.3*N, 5, 35], aspect='auto')
plt.title("matrix M error")
plt.xlabel("Outlier numbers")
plt.ylabel("Rank")
plt.colorbar()
plt.show()
