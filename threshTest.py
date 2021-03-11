import numpy as np
import matplotlib.pyplot as plt
from svt import svt
from st import st

m = 100
n = 50

## test st
x = np.ones(n)
y = np.linspace(-1,1,m)
tau = 0.5

# generate matrix and threshold
A = np.outer(x, y)
As = st(A, tau)

# display results
plt.figure()
plt.imshow(A, extent=[-1,1,0,n], aspect='auto')
plt.clim(-1,1)
plt.set_cmap('gray')
plt.colorbar()
plt.title('original matrix')
plt.show()

plt.figure()
plt.imshow(As, extent=[-1,1,0,n], aspect='auto')
plt.clim(-1,1)
plt.set_cmap('gray')
plt.colorbar()
plt.title('soft-thresholded matrix')
plt.show()

plt.figure()
plt.plot(A[0,:], label='original row')
plt.plot(As[0,:], label='soft-thresholded row')
plt.legend()
plt.show()

## test svt.m
D = 100
N = 500
r = 13
tau = 50

# generate matrix and threshold
A = np.random.randn(D,r) @ np.random.randn(r,N) + np.random.randn(D,N)
As = svt(A, tau)

# display results
plt.figure()
_, s, _ = np.linalg.svd(A)
plt.stem(s)
plt.title('$\sigma(A)$')
plt.show()

plt.figure()
_, s, _ = np.linalg.svd(As)
plt.stem(s)
plt.title('$\sigma(svt(A))$')
plt.show()
