import numpy as np

# Feel free to change the dimension parameters
# d, m, n, X, and Y below
d = 3
m = 4
n = 5

X = np.random.randn(d, m)
Y = np.random.randn(d, n)
# Solution 1: two-fold for-loop

D = np.zeros((m, n))
for j in range(n):
    for i in range(m):
        D[i, j] = np.linalg.norm(X[:, i] - Y[:, j])


# Solution 2: one-fold for-loop
D = np.zeros((m, n))
for j in range(n):
    Z = (X.T - Y[:, j]).T
    D[:, j] = sum(Z**2)
D = np.sqrt(D)

# Solution 3: no for-loop (the best solution).
# Using the hint. There are three terms, each can be computed
# for all m × n pairs in batch without for-loop.

tx2 = np.sum(X**2, 0)
ty2 = np.sum(Y**2, 0)
Txy = np.dot(X.T, Y)
D = tx2[:, np.newaxis] + ty2[np.newaxis, :] - 2 * Txy
D = np.sqrt(D)

# Simple benchmark shows that solution 3 is 10x to 30x
# faster than solution 1 for moderate size problems.
