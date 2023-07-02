import numpy as np

# Feel free to change the parameters
# n, C, and a below

n = 3
C = np.random.randn(n, n)
a = np.random.randn(1)


# Solution 1:
def regularization1(C, a):
    n = np.shape(C)[0]
    return C + a * np.eye(n)


# Solution 2:
def regularization2(C, a):
    n = np.shape(C)[0]
    idx = np.ravel_multi_index((range(n), range(n)), C.shape)
    C.flat[idx] += a
    return C


# While a little bit more complicated,
# solution 2 is much more efficient than solution 1 when $n$ is large.

print(regularization1(C, a))
print(regularization2(C, a))
