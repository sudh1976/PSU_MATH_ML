"""
RPCA demo - op.py

ECE 510

python version: 3.7.3

Spring 2019
"""

import numpy as np
from numpy.linalg import norm
from numpy import sqrt, zeros, sign


def op(M, test):
    r"""
    Principal Component Pursuit solved with ADMM

    Syntax:     L, S = pcp(M)

    Inputs:
        :param M: M matrix of size [m, n]
        :param omega: sampling set, Omega(i, j) = 1 if M_i,j is observed
        :param lam: input parameter

    Ouputs:
        L: The low rank matrix with size [D, N]
        S: The sparse matrix with size [D, N]
    """
### Parameters that we'll use

    m, n = np.shape(M)

    # choose any lambda smaller than (3 / 7) * 1/sqrt(0.05 * n)
    lam = 0.01      # ideal setup for mnist
    #lam = 0.01     # ideal setup for synthetic data
    #lam_s = 0.00099

    if test == 'mnist':
        lam = 0.01
        lam_s = 0.01
    else:
        lam = 0.008
        lam_s = 0.006

    # lam = 1 / sqrt(max(m, n))
    # lam = (3 / 7) * 1/sqrt(0.05 * n)

    delta = 0.00002
    mu_temp = 0.99 * norm(M, ord=1)
    mu_bar = delta * mu_temp
    eta = 0.9
    tol = 0.000001 * norm(M, ord='fro')
    stopping = 0

    # L_temp0, C_temp0 is L_k and C_k
    # L_temp1, C_temp1 is L_{k - 1} and C_{k - 1}

    L_temp0 = zeros((m, n))
    L_temp1 = zeros((m, n))
    C_temp0 = zeros((m, n))
    C_temp1 = zeros((m, n))
    t_temp0 = 1
    t_temp1 = 1
    k = 0

    while stopping != 1:
        #print(mu_temp)
        YL = L_temp0 + (t_temp1 - 1)/t_temp0*(L_temp0 - L_temp1)
        YC = C_temp0 + (t_temp1 - 1)/t_temp0*(C_temp0 - C_temp1)

        M_difference = (YL + YC - M)

        GL = YL - 0.5*M_difference
        L_new = svt(GL, mu_temp * lam/2)

        GC = YC - 0.5 * M_difference
        C_new = col_st(GC, mu_temp * lam_s/2)

        t_new = (1+sqrt(4*t_temp0**2+1))/2
        mu_new = max(eta*mu_temp, mu_bar)

        S_L = 2*(YL - L_new) + (L_new + C_new - YL - YC)
        S_C = 2*(YC - C_new) + (L_new + C_new - YL - YC)

        #print(f"S_L: {norm(S_L, ord='fro')**2}, S_C: {norm(S_C, ord='fro')**2}")
        # convergence condition
        err = norm(S_L, ord='fro')**2 + norm(S_C, ord='fro')**2
        print(f"{k}th trial - Error: {err}")
        if err < tol**2:
            stopping = 1
        else:
            L_temp1 = L_temp0
            L_temp0 = L_new
            C_temp1 = C_temp0
            C_temp0 = C_new
            t_temp1 = t_temp0
            t_temp0 = t_new
            mu_temp = mu_new
            k = k + 1

    return L_new, C_new


def st(X, tau):
    r"""
    Soft - thresholding/shrinkage operator

    Syntax:     Xs = st(X, tau)

    Input:
    :param X: Input matrix
    :param tau: Shrinkage parameter

    Output:
    Xs: The result of applying soft thrsholding to every element in X
    """
    Xtemp = abs(X) - tau
    Xtemp[Xtemp < 0.0] = 0.0
    Xs = sign(X) * Xtemp

    return Xs


def svt(X, tau):
    r"""
    Singular value thresholding operator

    Syntax:     Xs = svt(X, tau)

    Inputs:
        :param X: The input matrix
        :param tau: The input shrinkage parameter

    Output:
        Xs is the result of applying singular value thresholding to X
    """
    U, S, V = np.linalg.svd(X, full_matrices=False)
    Xs = U @ st(np.diag(S), tau) @ V

    return Xs


def col_st(C, epsilon):
    m, n = np.shape(C)
    output = np.zeros_like(C)
    for i in range(n):
        temp = C[:, i]
        norm_temp = norm(temp)
        if norm_temp > epsilon:
            temp = temp - temp*epsilon/norm_temp
        else:
            temp = zeros((m, 1))
        output[:, i] = np.squeeze(temp)

    return output