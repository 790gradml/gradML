import numpy as np

# Feel free to change the parameters
# n and A below
n = 3
A = np.random.randn(n, n)

# Creating a mask for the off-diagonal indices
msk = np.eye(n) == 0
v = np.max(A[msk])
