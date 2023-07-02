import numpy as np

# Feel free to change the parameters
# n, c, S, and X below

n = 4
c = np.random.randn(n)
# we need a PD covariance matrix S,
# here's one (common) way of creating a random PD matrix
_ = np.random.randn(n, n)
S = (_) @ (_.T) + 1e-3 * np.eye(n)
X = np.random.randn(n, n)

# Solution 1: naive solution as baseline
# Not good: inversing S for n times.
D = np.zeros(n)
for i in range(n):
    z = X[:, i] - c
    D[i] = np.dot(np.dot(z.T, np.linalg.inv(S)), z)

# Solution 2: do pre-computation
D = np.zeros(n)
invS = np.linalg.inv(S)
for i in range(n):
    z = X[:, i] - c
    D[i] = np.dot(np.dot(z.T, invS), z)

# Solution 3: vectorization
Z = X - c[:, np.newaxis]
invS = np.linalg.inv(S)
D = np.sum(Z.conj() * (np.dot(invS, Z)), axis=0)

# Solution 4: directly solving linear equations is
# more efﬁcient than doing inverse.
Z = X - c[:, np.newaxis]
D = np.sum(Z.conj() * (np.linalg.solve(S, Z)), axis=0)
