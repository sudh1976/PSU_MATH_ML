"""
RPCA demo - syntheticTest

ECE 510

python version 3.7.3

Spring 2019
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
D = 400
N = 400
r = 15
gamma = 0.05

A = randn(D, r)
B = randn(int(N - gamma * N), r)
C = randn(D, int(gamma * N))

Ltrue = np.hstack((A @ B.T, np.zeros((D, int(gamma*N)))))      # low - rank matrix
Strue = np.hstack((np.zeros((D, int(N-gamma*N))), C))    # sparse matrix
Y = Ltrue + Strue

# Utrue, Sigmatrue, Vtrue = svd(Ltrue)


# [Lpcp, Spcp] = pcp(Y)
[L, S] = op(Y)

errL = norm(Ltrue - L, ord='fro') / norm(Ltrue, ord='fro')
errS = norm(Strue - S, ord='fro') / norm(Strue, ord='fro')

print(f"error in L = {errL}")
print(f"error in S = {errS}")
