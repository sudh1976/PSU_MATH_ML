"""
RPCA demo - syntheticTest

ECE 510

python version 3.7.3

Spring 2019
"""

import numpy as np
from numpy.random import randn, rand
from numpy.linalg import norm
from pcp import pcp

## problem parameters
D = 100
N = 500
r = 13

Ltrue = randn(D, r) @ randn(r, N)       # low - rank matrix
Strue = np.float_(rand(D, N) < 0.01)    # sparse matrix
Y = Ltrue + Strue

[L, S] = pcp(Y)

errL = norm(Ltrue - L, ord='fro') / norm(Ltrue, ord='fro')
errS = norm(Strue - S, ord='fro') / norm(Strue, ord='fro')

print(f"error in L = {errL}")
print(f"error in S = {errS}")
