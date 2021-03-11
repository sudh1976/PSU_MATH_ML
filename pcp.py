import numpy as np
import numpy.linalg as la

def st(S, epsilon):
    r"""
    Soft thresholding

    Syntax:     Cs = cst(X, epsilon)

    Input:
    :param S: Input matrix
    :param epsilon: Shrinkage parameter

    Output:
    Ss: The result of applying soft thresholding to each column in S
    """

    M, N = S.shape
    Ss = np.zeros((M, N))

    for i in range(N):
        if np.absolute(S[i, i]) > epsilon:
            Ss[i, i] = S[i, i] - epsilon * np.sign(S[i, i])

    return Ss

def cst(C, epsilon):
    r"""
    Column Soft thresholding

    Syntax:     Cs = cst(X, epsilon)

    Input:
    :param C: Input matrix
    :param epsilon: Shrinkage parameter

    Output:
    Cs: The result of applying soft thresholding to each column in C
    """

    M, N = C.shape
    Cs = np.zeros((M, N))

    for i in range(N):
        Ci = C[:, i];
        norm_Ci = la.norm(Ci, ord=2);
        if norm_Ci > epsilon:
            Cs[:, i] = Ci - Ci * epsilon / norm_Ci;

    return Cs

def svt(X, epsilon):
    r"""
    Singular value thresholding operator

    Syntax:     Xs = svt(X, tau)

    Inputs:
        :param X: The input matrix
        :param epsilon: The input shrinkage parameter

    Output:
        Xs is the result of applying singular value thresholding to X
    """
    U, S, V = la.svd(X, full_matrices=False)
    Xs = U @ st(np.diag(S), epsilon) @ V

    return Xs



def pcp(M):
    r"""
    PCA Pursuit

    Syntax:     L, S = pcp(M)

    Inputs:     
        :param M: A matrix of size [D, N]

    Ouputs:
        L: The low rank matrix with size [D, N]
        C: The sparse matrix with size [D, N]
    """
### Parameters that we'll use
    D, N = np.shape(M)
    normM = la.norm(M, ord='fro')

### Algorithm parameters
    lam = 10 ** -5
    delta = 10 ** -5
    eta = 0.9
    mu = 0.99 * normM
    #tol = la.norm(M, ord='fro') * 1e-4
    tol = 1e-4

    maxIter = 1000

    print(tol)

### Initialize the variables of optimization
    L = L_prev = np.zeros([D, N])
    C = C_prev = np.zeros([D, N])
    t = t_prev = 1
    mu_ = delta * mu

    for ii in range(maxIter):
        Y_L = L + ((t_prev - 1.0) / t) * (L - L_prev)
        Y_C = C + ((t_prev - 1.0) / t) * (C - C_prev)

        G_L = Y_L - 0.5 * (Y_L + Y_C - M)
        G_C = Y_C - 0.5 * (Y_L + Y_C - M)

        L = svt(G_L, mu / 2)
        C = cst(G_C, lam * mu / 2.0)

        t = (1 + np.sqrt(4 * (t ** 2) + 1)) / 2
        mu = np.maximum(eta * mu, mu_)

## Calculate error and output at each iteration
# Stopping condition from the paper code
#         S_L = 2 * (Y_L - L) + (L + C - Y_L - Y_C)
#         S_C = 2 * (Y_C - C) + (L + C - Y_L - Y_C)
#         err = (la.norm(S_L, ord='fro')) ** 2  + (la.norm(S_C, ord='fro')) ** 2
        err = M - L - C
        err = la.norm(err, ord='fro') / normM
        print(f"Error: {err}")

        if err < tol:
            break

        #
        # if err <= tol ** 2:
        #     break

    return L, C

