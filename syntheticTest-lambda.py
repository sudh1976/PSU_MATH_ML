"""
RPCA demo - syntheticTest
"""

import numpy as np
from numpy.random import randn, rand
from numpy.linalg import norm
from op import op
import matplotlib.pyplot as plt


## problem parameters from Robust PCA
D = 400
N = 400
r = 15
gamma = 0.05

A = randn(D, r)
B = randn(int(N - gamma * N), r)
C = randn(D, int(gamma * N))

Ltrue = np.hstack((A @ B.T, np.zeros((D, int(gamma * N)))))  # low - rank matrix
Strue = np.hstack((np.zeros((D, int(N - gamma * N))), C))  # sparse matrix
Y = Ltrue + Strue

## Performance vs. Lambda
lam_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003]
errL_list = []
errS_list = []
for lam in lam_list:
    print(lam)
    L, S, _ = op(Y, test='synth', lam=lam, lam_s=lam)
    errL = norm(Ltrue - L, ord='fro') / norm(Ltrue, ord='fro')
    errS = norm(Strue - S, ord='fro') / norm(Strue, ord='fro')
    errL_list.append(errL)
    errS_list.append(errS)

plt.scatter(lam_list, errL_list, label="error on L")
plt.scatter(lam_list, errS_list, label="error on C")
plt.ylabel("error")
plt.xlabel(r'$\lambda$')
plt.legend()
plt.grid()
plt.show()
