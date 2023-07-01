# Programming problems

Try your best to avoid writing your own `for` loops.

## Regularization

Given an $n \times n$ matrix $C$, add a scalar $a$ to each diagonal
entry of $C$.

Solution:

```python
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
```

## Largest Off-diagonal Element

Given an $n \times n$ matrix $A$, find the value of the largest
off-diagonal element.


```py
import numpy as np

# Feel free to change the parameters
# n and A below
n = 3
A = np.random.randn(n, n)

# Creating a mask for the off-diagonal indices
msk = np.eye(n) == 0
v = np.max(A[msk])
```


## Pairwise Computation

Given a vector $x$ of length $m$, and a vector $y$ of length $n$,
compute $m \times n$ matrices: $A$ and $B$, such that
$A(i, j)=x(i)+y(j)$, and $B(i, j)=x(i) \cdot y(j)$.


```py
import numpy as np

# feel free to change the parameters
# m, n, x, and y below

m = 3
n = 4
x = np.random.randn(m)
y = np.random.randn(n)
A = x[:, np.newaxis] + y[np.newaxis, :]
B = x[:, np.newaxis] * y[np.newaxis, :]
```

## Pairwise Euclidean Distances

Given a $d \times m$ matrix $X$, and a $d \times n$ matrix $Y$, compute
an $m \times n$ matrix $D$, such that $D(i, j)=$
$\left\|x^{i}-y^{j}\right\|^{2}$, where $x^{i}$ is the $i$-th column of
$X$, and $y^{j}$ is the $j$-th column of $Y$. Hint: You may find the
following decomposition helpful for improving your code efficiency:

$$\left\|x^{i}-y^{j}\right\|^{2}=\sum_{k=1}^{d}\left(x_{k}^{i}-y_{k}^{j}\right)^{2}=\sum_{k=1}^{d} x_{k}^{i^{2}}+\sum_{k=1}^{d} y_{k}^{j^{2}}-\sum_{k=1}^{d} 2 x_{k}^{i} y_{k}^{j} .$$


```py
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
```


## Compute Mahalanobis Distances

The Mahalanobis distance is a measure of the distance between a point
$P$ and a distribution $D$, introduced by P. C. Mahalanobis in 1936. It
is a multi-dimensional generalization of the idea of measuring how many
standard deviations away ${P}$ is from the mean of ${D}$. This distance
is zero if ${P}$ is at the mean of ${D}$, and grows as ${P}$ moves away
from the mean: Along each principal component axis, it measures the
number of standard deviations from ${P}$ to the mean of ${D}$. If each
of these axes is rescaled to have unit variance, then Mahalanobis
distance corresponds to standard Euclidean distance in the transformed
space. Mahalanobis distance is thus unitless and scale-invariant, and
takes into account the correlations of the data set (from
<http://en.wikipedia.org/wiki/Mahalanobis_distance>). Given a center
vector $c$, a positive-definite covariance matrix $S$, and a set of $n$
vectors as columns in matrix $X$, compute the distances of each column
in $X$ to $c$, using the following formula:

$$D(i)=\left(x^{i}-c\right)^{T} S^{-1}\left(x^{i}-c\right) .$$Here, $D$
is a vector of length $n$.


```py
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
```

## 2-D Gaussian

Generate 1000 random points from a 2-D Gaussian distribution with mean
$\mu=[4,2]$ and covariance $$\Sigma=\left(\begin{array}{cc}
        1   & 1.5 \\
        1.5 & 3
    \end{array}\right)$$Plot the points so obtained, and estimate their
mean and covariance from the data. Find the eigenvectors of the
covariance matrix and plot them centered at the sample mean.


```py
import matplotlib.pyplot as plt
import numpy as np

N = 1000
mu = [4, 2]
sigma = [[1, 1.5], [1.5, 3]]


R = np.random.multivariate_normal(mu, sigma, N)
plt.plot(*(zip(*R)), marker=".", ls="")
plt.show()
muhat = np.mean(R, 0)
R_0 = R - muhat[np.newaxis, :]
SIGMA_hat = np.dot(R_0.T, R_0) / (N - 1)

L, Q = np.linalg.eig(SIGMA_hat)

plt.arrow(
    muhat[0],
    muhat[1],
    Q[0, 0],
    Q[1, 0],
    shape="full",
    lw=3,
    length_includes_head=True,
    head_width=0.01,
)
plt.arrow(
    muhat[0],
    muhat[1],
    Q[0, 1],
    Q[1, 1],
    shape="full",
    lw=3,
    length_includes_head=True,
    head_width=0.01,
)

plt.plot(*(zip(*R)), marker=".", ls="")
plt.axis([1, 6, -2, 6])
plt.show()
```


## Tournament fun

A tennis tournament starts with sixteen players. Let’s call them
$h_i, i=1,2, \dots 16$ (human $i$, to avoid the potentially confusing
notation $p_i$). The first round has eight games, randomly drawn/paired;
i.e., every player has an equal chance of facing any other player. The
eight winners enter the next round.

As an enthusiastic tennis and data fan, you have an internal model of
these 16 players based on their past performance. In particular, you
view each player $h_i$ as having a performance index score
$s_i ~ \sim \text{Gaussian}(\theta_i, \sigma_i^2).$ The mean $\theta_i$
roughly captures the player’s ‘intrinsic ability’ and the variance
$\sigma_i^2$ roughly captures the player’s performance reliability
(accounting for recent injuries etc.). In a match between $h_i$ and
$h_j$, player $h_i$ wins if $s_i>s_j$.

Based on your model, what’s the probability that your “top seed player”
(the one with the highest $\theta$) enters the next round? Run 10,000
simulations to check if it agrees with your answer.


Solution:

Suppose the top seed player is $h_1$. For $h_1$ to win against an
opponent $h_j$, we need the event $s_j-s_1 < 0$. Since $s_i$ and $s_j$
are independent normal distributions, their difference is also a normal
distribution, with mean $\theta_j - \theta_1$ and variance
$\sigma_j^2 + \sigma_1^2.$

Because $h_1$ has a $1/15$ chance of facing any one (and only one) other
player, we have 15 such disjoint events. So the total probability of
$h_1$ entering the next game is simply:
$$\frac{1}{15} \sum_{j=2}^{15} p (s_j-s_1 < 0)$$ where
$(s_j-s_1) \sim \text{Gaussian}(\theta_j- \theta_1, \sigma_j^2 + \sigma_1^2).$


Solution code below.

```py
import numpy as np
from scipy.stats import norm

# Feel free to change the parameters below
theta = np.linspace(3, 16, 16)
sigma = np.linspace(1, 2, 16)

# We'll start by getting our CDF solution

# get our top seed player's parameters
top_seed_index = np.argmax(theta)
top_seed_theta = theta[top_seed_index]
top_seed_sigma = sigma[top_seed_index]
# get the fifteen 'difference' random variable parameters
all_other_theta = np.delete(theta, top_seed_index)
all_other_sigma = np.delete(sigma, top_seed_index)
x_mean = all_other_theta - top_seed_theta
x_var = top_seed_sigma**2 + all_other_sigma**2
# x is the array holding the prob. of top seed player
# winning against each of the 15 opponents
x = [norm.cdf(0, loc=i, scale=np.sqrt(j)) for (i, j) in zip(x_mean, x_var)]
ans = np.sum(x) / 15
print(f"Top seed player's chance of winning is {ans}")

# We then run some simulations to see if the proportion of
# wins agree with the solution above
M = int(1e5)
count = 0


def one_simulation(all_other_theta, all_other_sigma):
    # choose a random opponent index
    j = np.random.choice(range(15))
    sj = norm.rvs(all_other_theta[j], all_other_sigma[j])
    top_seed_s = norm.rvs(top_seed_theta, top_seed_sigma)
    if top_seed_s < sj:
        return False
    return True


for i in range(M):
    if one_simulation(all_other_theta, all_other_sigma):
        count += 1
print(f"Top seed player wins {count/M} of the total simulated games.")
```