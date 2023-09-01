---
# layout: page
title: Background Review
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

## The Chain Rule (in calculus)
##  Positive Semidefiniteness (PSD) and Positive Definiteness (PD)

##  Convexity and Strong Convexity

{: .example}

Quadratic functions' Convexity Property

<iframe src="/demos/QuadraticFun.html" width=1000 height=800 async></iframe>

##  Optimal Solutions and Uniqueness

# Probability and Statistics


## Concepts related to a single distribution

### Basics and formulae
  - The notions of random variable, expectation, variance, entropy.
  - Union bounds
  - Concentration inequalities: Markov's Inequality, Chernoff bound, Hoeffding's inequality

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
###  Marginal Independence

Let $(\Omega, \mathcal{F} \cdot \mathbb{P})$ be a probability space. Two events, $A$ and $B$, are said to be independent if $\mathbb{P}(A \cap B)=\mathbb{P}(A) \mathbb{P}(B)$. If $\mathbb{P}(B)>0$, an equivalent condition is $\mathbb{P}(A)=\mathbb{P}(A \mid B)$.

Note: As an immediate consequence of that joint probability is the product of individual probabilities, the entropy  of the joint distribution is the sum of individual entropies

### Bayes' Rule

Let $A$ be an event with $\mathbb{P}(A)>0$. If the events $B_i, i \in \mathbb{N}$, form a partition of $\Omega$, and $\mathbb{P}\left(B_i\right)>0$ for every $i$, then

$$
\mathbb{P}\left(B_i \mid A\right)=\frac{\mathbb{P}\left(B_i\right) \mathbb{P}\left(A \mid B_i\right)}{\mathbb{P}(A)}=\frac{\mathbb{P}\left(B_i\right) \mathbb{P}\left(A \mid B_i\right)}{\sum_{j=1}^{\infty} \mathbb{P}\left(B_j\right) \mathbb{P}\left(A \mid B_j\right)}
$$

### Conditional Independence
  - Comparison with marginal independence

### Importance sampling


### KL divergence