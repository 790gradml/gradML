# Programming problems

Try your best to avoid writing your own `for` loops.

## Regularization

Given an $n \times n$ matrix $C$, add a scalar $a$ to each diagonal
entry of $C$.

Solution:

```py
{% include_relative /code/Regularization.py %}
```


## Largest Off-diagonal Element

Given an $n \times n$ matrix $A$, find the value of the largest
off-diagonal element.

```py
{% include_relative /code/Largest_offdiagonal.py %}
```


## Pairwise Computation

Given a vector $x$ of length $m$, and a vector $y$ of length $n$,
compute $m \times n$ matrices: $A$ and $B$, such that
$A(i, j)=x(i)+y(j)$, and $B(i, j)=x(i) \cdot y(j)$.

```py
{% include_relative /code/pairwise.py %}
```

## Pairwise Euclidean Distances

Given a $d \times m$ matrix $X$, and a $d \times n$ matrix $Y$, compute an $m \times n$ matrix $D$, such that $$D(i, j)=\left\|x^{i}-y^{j}\right\|$$, where $x^{i}$ is the $i$-th column of $X$, and $y^{j}$ is the $j$-th column of $Y$. Hint: You may find the following decomposition (of the norm/distance squared) helpful for improving your code efficiency:

$$\left\|x^{i}-y^{j}\right\|^{2}=\sum_{k=1}^{d}\left(x_{k}^{i}-y_{k}^{j}\right)^{2}=\sum_{k=1}^{d} x_{k}^{i^{2}}+\sum_{k=1}^{d} y_{k}^{j^{2}}-\sum_{k=1}^{d} 2 x_{k}^{i} y_{k}^{j} .$$

```py
{% include_relative /code/pairwise_distance.py %}
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
[wikipedia](http://en.wikipedia.org/wiki/Mahalanobis_distance)). Given a center
vector $c$, a positive-definite covariance matrix $S$, and a set of $n$
vectors as columns in matrix $X$, compute the distances of each column
in $X$ to $c$, using the following formula:

$$D(i)=\left(x^{i}-c\right)^{T} S^{-1}\left(x^{i}-c\right) .$$Here, $D$
is a vector of length $n$.

```py
{% include_relative /code/mahalanobis.py %}
```

## 2-D Gaussian

Generate 1000 random points from a 2-D Gaussian distribution with mean
$\mu=[4,2]$ and covariance 

$$\Sigma=\left(\begin{array}{cc}
        1   & 1.5 \\
        1.5 & 3
    \end{array}\right)$$
    
Plot the points so obtained, and estimate their mean and covariance from the data. Find the eigenvectors of the
covariance matrix and plot them centered at the sample mean.


```py
{% include_relative /code/2DGaussian.py %}
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
{% include_relative /code/tournament.py %}
```