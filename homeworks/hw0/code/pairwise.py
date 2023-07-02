import numpy as np

# feel free to change the parameters
# m, n, x, and y below

m = 3
n = 4
x = np.random.randn(m)
y = np.random.randn(n)
A = x[:, np.newaxis] + y[np.newaxis, :]
B = x[:, np.newaxis] * y[np.newaxis, :]
