"""
RPCA demo - syntheticTest
"""

import numpy as np
from numpy.random import randn, rand
from numpy.linalg import norm, svd
from pcp import pcp
from op import op


## problem parameters from Robust PCA
# D = 100
# N = 500
# r = 13
#
# Ltrue = randn(D, r) @ randn(r, N)       # low - rank matrix
# Strue = np.float_(rand(D, N) < 0.01)    # sparse matrix
# Y = Ltrue + Strue

## problem parameters from Outlier Pursuit paper
D = np.linspace(100, 1000, 10).astype(int)
N = np.linspace(100, 1000, 10).astype(int)
r = 15
gamma = 0.05

T = np.zeros(len(D))

for d in range(len(D)):
    A = randn(D[d], r)
    B = randn(int(N[d] - gamma * N[d]), r)
    C = randn(D[d], int(gamma * N[d]))

    Ltrue = np.hstack((A @ B.T, np.zeros((D[d], int(gamma*N[d])))))      # low - rank matrix
    Strue = np.hstack((np.zeros((D[d], int(N[d]-gamma*N[d]))), C))    # sparse matrix
    Y = Ltrue + Strue

    # Utrue, Sigmatrue, Vtrue = svd(Ltrue)

    # [Lpcp, Spcp] = pcp(Y)
    L, S, T[d] = op(Y, test='synthetic', lam=0.008, lam_s=0.006)


    # Commented out for time plot
    #errL = norm(Ltrue - L, ord='fro') / norm(Ltrue, ord='fro')
    #errS = norm(Strue - S, ord='fro') / norm(Strue, ord='fro')

    #print(f"error in L = {errL}")
    #print(f"error in S = {errS}")

plt.stem(T, D)
plt.title('Run Time vs Problem Size')
plt.ylabel('Problem Size ($N \by N$)')
plt.xlabel('Run Time (s)')
plt.savefig('runtime_vs_probsize')

plt.show()