---
# layout: page
title: Background/Review
nav_order: 3
# katex: true
---

# Backrgound and Review
{: .no_toc}

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

## Notations

- Data matrix is of the size $(n,d)$ where $n$ is the number of data points, and $d$ is the dimension of the features

- Vectors are denoted with a small-case letter; matrices capital letters
- The default norm of a vector is the $l_2$ norm

## Linear Algebra, Calculus, and Optimization

- Gradient vector
- The chain rule (in calculus)
- Positive semidefiniteness (PSD) and positive definiteness (PD)
- Convexity, strong convexity
- Optimal solutions, uniqueness

## Probability and Statistics

### Concepts related to a single distribution

- Basics and formulas
- Fundamental distributions
  - 1d normal
  - Multi-variate normal
- Jensen's inequality
- Max likelihood and Max log likelihood

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
such reasons, optimization is an integral part of Machine Learning.

The ML estimation for variance (and standard deviation) is biased. This
leads to the Bessel correction for variance:
$$\tilde{\sigma}^2_{\rm ML}=\frac{1}{N-1}\sum_{i=1}^N (x_i-\mu_{\rm ML})^2.$$

### Concepts involving multiple distributions

- The chain rule (in probability)
- Marginal independence
  - Joint probability is the product
  - Entropy of the joint distribution is the sum of individual entropies
- Bayes' rule
- Conditional independence
  - Comparison with marginal independence
- Importance sampling
- KL divergence
