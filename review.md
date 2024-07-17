---
# layout: page
title: Review
nav_order: 2
# katex: true
---

# Background Review
{: .no_toc}

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

# Notations

- Data matrix is of the size $(n,d)$ where $n$ is the number of data points, and $d$ is the dimension of the features
- Vectors are denoted with a small-case letter; matrices capital letters
- The default norm of a vector is the $l_2$ norm


# Linear Algebra, Calculus, and Optimization

## The Gradient Vector

Consider a multi-variate function $f: \mathbb{R}^n \rightarrow \mathbb{R}.$ Assuming the function is differentiable at a point $p=\left(x_1, \ldots, x_n\right)$ in its $n$-dimensional input space, then the function's gradient, denoted as $\nabla f: \mathbb{R}^n \rightarrow \mathbb{R}^n,$ at that particular point $p$ is the $n$-dimensional vector

$$
\nabla f(p)=\left[\begin{array}{c}
\frac{\partial f}{\partial x_1}(p) \\
\vdots \\
\frac{\partial f}{\partial x_n}(p)
\end{array}\right]
$$


## The Chain Rule (in calculus)

##  Positive Semidefiniteness (PSD) and Positive Definiteness (PD)
Positive semidefinite(ness) and the variants are extremely important concepts in optimization. For example, they have direct implications on whether a point is a local minimum. 

For now, let us try to review these concepts on their own; starting with PSD. Recall that:

{: .definition}
An $n \times n$ symmetric real matrix $A$ is said to be positive semidefinite (i.e., $A \succeq 0$) if $x^{T} A x \geq 0$ for all $x$ in $\mathbb{R}^{n}$.

This definition requires checking the sign of the left-hand-side $x^{T} A x$ for all possible $x$ in $\mathbb{R}^{n}$ to establish $A \succeq 0$; not an easy task in general. Luckily there are many equivalent conditions that allow us to 
more efficiently/computationally check the PSD property. Specifically, the following are all equivalent:

- $A \succeq 0$
- All $2^{n}-1$ principal minors of $A$ are nonnegative. (The so-called Sylvester's criterion)
- All eigenvalues of $A$ are nonnegative.
- There exists a factorization $A=B^{T} B$.


The story is analogous for positive definiteness (PD). 

{: .definition}
An $n \times n$ symmetric real matrix $A$ is said to be positive definite (i.e., $A \succ 0$) if $x^{T} A x > 0$ for all $x$ in $\mathbb{R}^{n}$ and $x\neq 0$.


And similarly, the following are all equivalent conditions (some are more ``actionable'' than the definition):

- $A \succ 0$
- All $n$ **leading** principal minors of $A$ are positive. (Note that Sylvester's conditions for PD and PSD are majorly different)
- All eigenvalues of $A$ are positive. 
- There exists a factorization $A=B^{T} B$ where $B$ is square and non-singular.

We will skip the details for negative semi-definite ($A\preceq 0$) and negative definite ($A\prec 0$), because, e.g., $A\preceq 0$ if and only if $-A\succeq 0$ (convince yourself of this); so understanding the PSD/PD case is enough. 


##  Convexity and Strong Convexity
- Convexity definition
- Strong convexity definition
- If a function is strongly convex, it is convex.

{: .example}
Quadratic functions' Convexity Property

<iframe src="/demos/QuadraticFun.html" width=1000 height=800 async></iframe>

##  Optimal Solutions and Uniqueness

Some quick facts:
- If a function is convex, and this function has any local minima, then all of those local minima are global minima. Note that such local/global minima are not necessarily unique.
- If a function is strongly convex, and this function has any local minima, then the following are true:
  - any local minima must be unique, that is, there can only exist a single unique local minimum
  - this local minimum is a global minimum
  - this local minimum is a unique global minimum

# Probability and Statistics


## Concepts related to a single distribution

### Basics and formulae
  - The notions of random variable, expectation, variance, entropy.
  - Union bounds
  - Concentration inequalities: Markov's Inequality, Chernoff bound, Hoeffding's inequality
  - (Weak) law of large numbers

### Fundamental Distributions

**Discrete Distributions PMFs (Probability Mass Functions)**

- Discrete *uniform* with parameters $a$ and $b$, where $a$ and $b$ are integers with $a<b$. Here,
$$
p_X(k)=1 /(b-a+1), \quad k=a, a+1, \ldots, b,
$$
and $p_X(k)=0$, otherwise. (In the remaining examples, the qualification " $p_X(k)=0$, otherwise," will be omitted brevity.)

- *Bernoulli* with parameter $p$, where $0 \leq p \leq 1$. Here, $p_X(0)=p, p_X(1)=$ $1-p$.

- *Binomial* with parameters $n$ and $p$, where $n \in \mathbb{N}$ and $p \in[0,1]$. Here,

$$
p_X(k)=\left(\begin{array}{l}
n \\
k
\end{array}\right) p^k(1-p)^{n-k}, \quad k=0,1, \ldots, n
$$

**Continuous Distributions PDFs (Probability Density Functions)**
- 1D Gaussian
- Multivariate Gaussian

### Jensen's Inequality

<iframe src="/demos/Jensen.html" width=1000 height=800 async></iframe>

###  Max likelihood and Max log likelihood

{: .example}
MLE for Gaussian

We have data $x_1,\ldots,x_N$ sampled from a distribution. The goal is
to learn the distribution. The assumption is that the data is generated
from a Gaussian distribution $\mathcal{N}(\mu,\sigma^2)$. Then the
refined goal is to learn the mean and variance. How to learn
(parameters, mean and variance)?

A common method is maximum likelihood (ML), that is, choose the
parameters that maximize $\mathbb{P}(\text{data}|\text{parameters})$. In
this problem, to choose mean, variance from samples, the likelihood is

$$\begin{aligned}
    \mathbb{P}\left(x_1,\ldots,x_N|\mu,\sigma^2\right)=&\prod_{i=1}^N\mathbb{P}\left(x_i|\mu,\sigma^2\right) \\
    =&\prod_{i=1}^N \frac{1}{(2\pi\sigma^2)^{1/2}}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right).
\end{aligned}$$

Maximizing likelihood is same as maximizing logarithm of
likelihood. This leads to

$$\max_{\mu,\sigma^2} g(\mu,\sigma^2),$$

where $$g(\mu,\sigma^2)=-\frac{1}{2\sigma^2}\sum_{i=1}^N(x_i-\mu)^2-N\ln\sigma -N\ln\sqrt{2\pi}.$$
This is an optimization problem and its solution is what we desire. For
such reasons, optimization is an integral part of Machine Learning. The ML estimation for variance (and standard deviation) is biased. This
leads to the Bessel's correction for variance:

$$\tilde{\sigma}^2_{\rm ML}=\frac{1}{N-1}\sum_{i=1}^N (x_i-\mu_{\rm ML})^2.$$

## Concepts involving multiple distributions

### Conditional probability

Consider a probability space $(\Omega, \mathcal{F}, \mathbb{P})$, and an event $B \in \mathcal{F}$ with $\mathbb{P}(B)>0$. For every event $A \in \mathcal{F}$, the conditional probability that $A$ occurs given that $B$ occurs is denoted by $\mathbb{P}(A \mid B)$ and is defined by

$$
\mathbb{P}(A \mid B)=\frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}.
$$

###  The Chain Rule (in probability)
By reversing the definition of conditional probability (above), we arrive at the Chain Rule:

$$
\mathbb{P}(A \cap B)=\mathbb{P}(A \mid B)\mathbb{P}(B) 
$$

(which is also valid for when $\mathbb{P}(B)=0.$)



###  Marginal Independence

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space. Two events, $A$ and $B$, are said to be independent if

$$
\mathbb{P}(A \cap B)=\mathbb{P}(A) \mathbb{P}(B).
$$

If $\mathbb{P}(B)>0$, an equivalent condition is 

$$\mathbb{P}(A)=\mathbb{P}(A \mid B).$$

**Note**: Given marginal independence of two random variables, their joint distribution is the product of the individual marginal distribution. As an immediate consequence of this fact, given marginal independence of two random variables, the entropy of their joint distribution is equal to the sum of their individual entropies.



### Bayes' Rule

Let $A$ be an event with $\mathbb{P}(A)>0$. If the events $B_i, i \in \mathbb{N}$, form a partition of $\Omega$, and $\mathbb{P}\left(B_i\right)>0$ for every $i$, then

$$
\mathbb{P}\left(B_i \mid A\right)=\frac{\mathbb{P}\left(B_i\right) \mathbb{P}\left(A \mid B_i\right)}{\mathbb{P}(A)}=\frac{\mathbb{P}\left(B_i\right) \mathbb{P}\left(A \mid B_i\right)}{\sum_{j=1}^{\infty} \mathbb{P}\left(B_j\right) \mathbb{P}\left(A \mid B_j\right)}
$$

### Conditional Independence
  - Comparison with marginal independence

### Importance sampling


### KL divergence

KL divergence is a measure of the dissimilarity between two probability distributions; it quantifies how one probability distribution diverges from another. The quantity we're after, $$KL(P \| Q)$$, measures the information lost when using Q to approximate P. (In more technical information-theory language, it calculates the extra amount of information needed to encode data from P using a code optimized for Q.)

{: .definition}
> Given two discrete probability distributions $P$ and $Q$, the KL divergence between $P$ and $Q$ is defined as:
>
> $$KL(P \| Q)= - \sum_x P(x) \log \left(\frac{Q(x)}{P(x)}\right)$$
>

(for continuous random variables, the summation in the definition is to be replaced by the integral.)


Some useful properties of KL divergence:
  1. Non-negativity: $$KL(P \| Q) \geq 0$$, with equality if and only if P and Q are identical.
  2. Lack of symmetry: $$KL(P \| Q) \neq KL(Q \| P)$$ in general.
  <!-- 3. Invariance under change of variables: $$KL(P \| Q)$$ remains the same if the variables are transformed. -->
