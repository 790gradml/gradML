---
title: Discriminative vs Generative Classification
nav_order: 1
---

# Discriminative vs Generative Classification
{: .no_toc}
## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

# Overview

In this lecture, we will mainly discuss two different approaches to
build classifiers, the generative approach and the discriminative
approach. As concrete examples, we will look at the Naive Bayes
classifier for the generative approach and compare it with the logistic
regression, as an example of discriminative approach. We will show under
certain conditions, the Naive Bayes is a linear classifier, just as
Logistic Regression, but it assumes stronger assumptions and therefore
is a more biased classifier.

# Four Approaches to Build Classifiers

## Review: Bayes Classifier

We start our discussion of classification from the Bayes Classifier in
[Classification Fundamentals](/supervised/classification_fundamentals/). Recall the Bayes classifier $h^{*}(x)$ is defined by the
rule: 

$$h^{*}(x) :=
  \begin{cases}
    1, &\text{if}\ \eta(x)=\mathbb{P}(Y=1 | X=x)>\frac{1}{2}\\
    0, &\text{otherwise}.
  \end{cases}$$
  
The classifier predicts label $1$ if the conditional
probability of being in class $1$ is bigger than half. We also showed
this classifier is actually the optimal possible classifier, as the
underlying distribution $\mathbb{P}$ is assumed known.

## When Bayes Classifier is not feasible

However, we know this idealized situation is seldom the case in reality,
as we usually do not have access to $\mathbb{P}$. Therefore, we
introduced two different approaches, distance based classification (e.g.
Nearest Neighbors) and Empirical Risk Minimization (e.g. SVM). Here, we
give a formal summary of four possible methods to learn a classifier.

- Distance based method

- Empirical Risk Minimization

- Discriminative Approach: Fit a model $\mathbb{\hat{P}}(Y\|X)$ to
  approximate the conditional distribution $\mathbb{P}(Y|X)$, add
  the "classify" part using: $h(x)=\\argmax_{y}\mathbb{\hat{P}}(Y=y|X=x)$

- Generative Approach: Fit a model $\mathbb{\hat{P}}(X, Y)$ to
  approximate the joint distribution $\mathbb{P}(X, Y)$, add the "classify" part
  using: $$h(x)=\argmax_{y}\mathbb{P}(X=x,Y=y) =\argmax_{y} \mathbb{P}(X=x | Y=y)\mathbb{P}(Y=y)$$

# Discriminative vs. Generative

The discriminative and the generative methods are two approaches to
approximate the unknown underlying true distribution, and they re
related by the Bayers' rule $$\mathbb{P}(X,Y)=\mathbb{P}(X|Y)\mathbb{P}(Y)=\mathbb{P}(Y|X)\mathbb{P}(X)$$. Concretely,
the generative approach learn what the individual classes looks like and
models the data distribution $\mathbb{P}(X)$. It is a potentially harder problem
and computationally more challenging. But the advantage of this approach
is that it can be used to sample new data.

On the other hand, the discriminative approach learn the boundary
between classes and models $\mathbb{P}(Y|X)$, the conditional distribution and
ignores $\mathbb{P}(X)$. It solves a potentially easier problem and
computationally simpler. However, it cannot be used to sample new data.



# Discriminative Classifiers: Logistic Regression

Logistic Regression can be viewed as an approach of fitting a
discriminative model. It assumes a parametric form of the conditional
distribution $\mathbb{P}(Y|X)$ as
$$\mathbb{P}(Y=1|X=x;w) = \frac{e^{w^Tx}}{1+e^{w^Tx}} = \sigma (w^Tx)$$ where
$\sigma(z) = \frac{1}{1+e^{-z}}$ and is often called the Sigmoid
function.

Therefore,
$$\mathbb{P}(Y=0|X=x;w) = 1- \frac{e^{w^Tx}}{1+e^{w^Tx}} = 1- \sigma (w^Tx)$$ In
the training process, we are given a training data set
$S=\{(x_1,y_1), ... , (x_N,y_N)\}$ to estimate the weights $w$.
Naturally, we will apply the maximum likelihood method where we can
write the likelihood of the training data, assuming i.i.d., as

$$\begin{aligned}
\mathbb{P}(S|w) & = \prod_{i=1}^N \mathbb{P}(y_i|x_i;w) \\
    & = \prod_{i=1}^N ( \sigma (w^Tx_i))^{y_i} (1-\sigma (w^Tx_i) ) ^{1-y_i}
\end{aligned}$$

Equivalent to maximize $\mathbb{P}(S|w)$, we can minimize the
negative log-likelihood of $S$, which is

$$\begin{aligned}
L(w) &=  - \sum_{i=1}^N \log [(\sigma(w^Tx_i))^{y_i}(1-\sigma(w^Tx_i))^{1-y_i}] \\
& =  - \sum_{i=1}^N [y_i \log \sigma(w^Tx_i) + (1-y_i) \log (1-\sigma (w^Tx_i))]
\end{aligned}$$

This is often called the cross-entropy error.

# Generative Classifiers: Naive Bayes

Now we take a closer look at the generative approach. As said before, in
the generative approach, we fit a model of the joint distribution
$\mathbb{P}(X,Y)$ and derived our classifier using the Bayes rule:
$$\mathbb{P}(Y|X) = \frac{\mathbb{P}(X|Y)\mathbb{P}(Y)}{\mathbb{P}(X)}$$ where $$\mathbb{P}(X)=\sum_Y \mathbb{P}(X|Y)\mathbb{P}(Y)$$. 

For classification, we have 

$$\begin{aligned}
h(x)  & = \argmax_y \mathbb{P}(Y=y|X=x)\\
    & = \argmax_y \frac{\mathbb{P}(X=x|Y=y)\mathbb{P}(Y=y)}{\mathbb{P}(X=x)} \\
    & = \argmax_y \mathbb{P}(X=x|Y=y)\mathbb{P}(Y=y)
\end{aligned}$$ 

The denominator $\mathbb{P}(X=x)$ is only a normalization
constant and thus can be ignored for deriving $\argmax_y \mathbb{P}(Y=y|X=x)$.

Now let’s suppose both $X$ and $Y$ are discrete random variables, where
$X \in \mathcal{X}^d$ and $Y \in \mathcal{Y}$. Then we have
$$\mathbb{P}(Y=y_i|X=x_i) \propto \mathbb{P}(X=x_k|Y=y_i)\mathbb{P}(Y=y_i)$$ Remember $X$ is a $d$
dimensional random variable, so to fully express this conditional
distribution, we will approximately need
$|\mathcal{X}|^d |\mathcal{Y}| + |\mathcal{Y}|$ parameters. As a simple
example, suppose $$\mathcal{X}=\mathcal{Y}=\{0,1\}$$, then
$$X \in \{0,1\}^d$$. To specify $\mathbb{P}(X=x_k|Y=0)$, we need $2^d$ parameters.
Unless $d$ is sufficiently small, this full distribution is usually not
computational tractable.

## The Naive Bayes assumption

To make the computation tractable and make the problem simpler, the
Naive Bayes model make a **strong** assumption that the $d$ features are
conditional independent of each other given the class label $Y$:

$$
\begin{aligned}
\mathbb{P}(X|Y) &= \mathbb{P}(X_1,X_2,...,X_d|Y) \\
    &=\mathbb{P}(X_1|X_2,...X_d|Y)\mathbb{P}(X_2|X_3,...,X_d|Y)...\mathbb{P}(X_d|Y) \\
    &=\mathbb{P}(X_1|Y)\mathbb{P}(X_2|Y)...\mathbb{P}(X_d|Y)
\end{aligned}$$ 

where the second ’$=$’ used the conditional independence. Therefore, our classifier becomes
$$\argmax_y \mathbb{P}(Y=y|X_1,...,X_d) \propto \mathbb{P}(Y=y)\prod_{j=1}^d \mathbb{P}(X_j|Y=y).$$

Now, using maximum likelihood on training data $S$, we can estimate the
parameters for the Naive Bayes classifier, assuming a specific
distribution for $\mathbb{P}(X_j|Y)$, $1\leq j \leq d$. We can show for a
multinomial distribution of $\mathbb{P}(X_j|Y)$, the parameters of Naive Bayes
are 

$$\begin{aligned}
\mathbb{P}(Y=y) &= \frac{\# \text{Example with }Y = y}{N} \\
\mathbb{P}(X_i=x|Y=y) &= \frac{\# \text{Example with } X_i = x \text{ and } Y=y}{\# \text{Example with }Y = y}
\end{aligned}$$
## Naive Bayes can be a linear classifier

We can show that under some common assumptions on $\mathbb{P}(X_j|Y)$, the Naive
Bayes classifier is actually a linear classifier. Here we provide a
proof for when $$X_j \in \{0,1\}$$ and $\mathbb{P}(X_j|Y)$ is a Bernoulli distribution.
(See Exercise 2 Problem 2 for other more general situations.)

{: .theorem}
Assuming that $$X_j \in \{0,1\}$$, and $\mathbb{P}(X_j\|Y)$, $1\leq j\leq d$
is a Bernoulli distribution. The Naive Bayes classifier is defined by
$$h(x) = \text{sign} (w^Tx+w_0)$$ for a suitable choice of $w$,$w_0$.

{: .proof}
> Since $$X_j \in \{0,1\}$$, the Bernoulli distribution is
> 
> $$\begin{aligned}
\mathbb{P}(X_j|Y=1) & = a_j^{X_j} (1-a_j)^{(1-X_j)} \\
\mathbb{P}(X_j|Y=0) & = b_j^{X_j} (1-b_j)^{(1-X_j)}
\end{aligned}$$
> 
> where $a_j$ and $b_j$ are parameters for the $j$th
> dimension of $X$. With the conditional independence, we have 
> 
> $$\begin{aligned}
\mathbb{P}(Y=1|X)  & =\frac{\mathbb{P}(X|Y=1)\mathbb{P}(Y=1)}{\mathbb{P}(X|Y=1)\mathbb{P}(Y=1)+\mathbb{P}(X|Y=0)\mathbb{P}(Y=0)} \\
        & =\frac{1}{1+\frac{\mathbb{P}(X|Y=0)\mathbb{P}(Y=0)}{\mathbb{P}(X|Y=1)\mathbb{P}(Y=1)}} \\
        & = \frac{1}{1+\exp (- \log \frac{\mathbb{P}(X|Y=1)\mathbb{P}(Y=1)}{\mathbb{P}(X|Y=0)\mathbb{P}(Y=0)})} \\
        & =\sigma \left( \sum_j^d \log \frac{\mathbb{P}(X_j|Y=1)}{\mathbb{P}(X_j|Y=0)} + \log \frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)} \right)
\end{aligned}$$ 
> 
> and therefore,
> 
> $$\begin{aligned}
\mathbb{P}(Y=1|X)  & = \sigma \left( \sum_j^d \log \frac{a_j^{X_j}(1-a_j)^{(1-X_j)}}{b_j^{X_j}(1-b_j)^{(1-X_j)}} + \log \frac{p}{1-p} \right) \\
        & = \sigma \left( \sum_j^d \left(X_j \log \frac{a_j(1-b_j)}{b_j(1-a_j)}\right) + \log \left(\frac{p}{1-p} \prod_j^d \frac{1-a_j}{1-b_j} \right) \right) \\
        & = \sigma \left( \sum_j^d w_j X_j  + w_0\right)
\end{aligned}$$
> 
> where $w_j = \log \frac{a_j(1-b_j)}{b_j(1-a_j)}$ and
> $w_0=\log \left(\frac{p}{1-p} \prod_j^d \frac{1-a_j}{1-b_j} \right)$.
> 
> Therefore, $$h(x) = \text{sign} (w^Tx+w_0)$$ and this shows the Naive Bayes is a linear classifier.

# Naive Bayes vs. Logistic Regression

We saw in the last section that Naive Bayes can be a linear classifier
under some assumptions, but some of the assumptions are quite strong,
such as the conditional independence. Thus, the hypothesis class of
Naive Bayes is not all the possible linear classifiers, but only a
subset of them. We know the Logistic Regression is another common linear
classifier, how does Naive Bayes compare to Logistic Regression? In this
section, we will look at them more closely through a theoretic view.

## Asymptotic Regime

We first define certain notations for our discussion, let
$\epsilon(h_A,N) \equiv L(h_A, S)$, where $|S| = N$. Here,
$\epsilon(h_A,N)$ stands for the error of hypothesis $h$ trained using
algorithm $A$ from $N$ observations. We will first discuss in the
asymptotic setting, which means the number of training data points is
infinity.

If the two classes are linear separable, then we have
$$\epsilon(h_{NB}, \infty) = \epsilon(h_{LR},\infty)$$ which means,
asymptotically, NB and LR produce the identical classifier.

If the linear separable assumption does not hold, which is usually the case, we
claim LR is expected to outperform NB, i.e.
$$\epsilon(h_{LR},\infty) \leq \epsilon(h_{NB},\infty)$$ Intuitively,
this can be shown by observing that, since Logistic Regression assumes
nothing more than linear classification,
$\epsilon(h_{LR},\infty)$ converges to $inf_{h\in \mathcal{H}} L(h)$,
where $\mathcal{H}$ is the class of all linear classifiers, it must
therefore be asymptotically no worse than the linear classifier picked
by Naive Bayes, which assumes conditional independence between features.

## Non-asymptotic Regime

In a more real setting, where we do not have infinite number of training
data, we must talk about how fast (how many training sample needed) an
estimator converges to its asymptotic limit. This "rate of convergence"
for Logistic Regression is as below

{: .theorem}
Let $h_{LR,N}$ be a Logistic Regression model in $d$
dimensions. Then, with high probability,
$$\epsilon(h_{LR,N} \leq \epsilon(h_{LR},\infty)) + O \left( \sqrt{\frac{d}{N} \log \frac{N}{d}} \right)$$
Thus, for
$\epsilon(h_{LR}, N) \leq \epsilon(h_{LR},\infty) + \epsilon_0$ to hold
up for a fixed constant $\epsilon_0$, it suffices to pick $N=\Omega(d)$

The proof of this theorem follows the application of the
uniform convergence bounds to logistic regression, and using the fact
that the $d$ dimensional linear classifier $\mathcal{H}$ has a VC
dimension of $d+1$. These concepts will be covered in [Learnability and VC dimension](/supervised/learnability_and_vc).

Now, we want to draw a similar picture for the Naive Bayes classifier
and compare it the Logistic Regression. It turns out this is a more
challenging task and we will break our analysis into two parts

- How fast do parameters of NB converge to their optimal values

- How fast do the risk if NB converge to the asymptotic risk

For the first part, formally, we have the following theorem:

{: .theorem}
> Let any $\epsilon_1$, $\delta >0$ and any $l \leq 0$
> be fixed. Assume that for some fixed $\rho_0 > 0$, we hvae
> $\rho_0 \leq \mathbb{P}(y=1) \leq 1- \rho_0$. Let
> $d = O ((1/\epsilon_1^2)\log(d/\delta))$, then with probability at least
> $1-\delta$:
> 
> $$\begin{aligned}
  |\hat{P}(X_j|Y=b) - \mathbb{P}(X_j|Y=b)| \leq \epsilon_1  
\end{aligned}$$
> 
> and
> 
> $$\begin{aligned}
  |\hat{P}(Y=b) - \mathbb{P}(Y=b)| \leq \epsilon_1
\end{aligned}$$ for all $j=1,...,d$ and $b \in \mathcal{Y}$

This theorem states that with a number of samples that is only
*logarithmic*, rather than linear, in $d$, the parameters of Naive Bayes
are uniformly close to their asymptotic values in $h_{NB,\infty}$.

Proof of this theorem is a straightforward application of the Hoeffding
bound, here we provide a simple setting where $$X \in \{0,1 \}$$ and
$\mathbb{P}(X=1)=p$. Suppose we have $N$ i.i.d. samples $(x_1,...,x_N)$, then the
maximum likelihood estimation for $p$ is simply
$$\hat{p} = \frac{1}{N}\sum_i x_i.$$ 

The Hoeffding bound for this case
states, for all $p \in [0,1]$, $\epsilon >0$
$$\mathbb{P}(|p-\hat{p}|>\epsilon) \leq 2 e^{-2N\epsilon^2}$$ Intuitively, this
means that the probability of the empirical estimation being epsilon-far
from the ground truth decays exponentially fast with the number of
samples $N$. For a detailed proof of theorem
<a href="#thm:NB_convergence" data-reference-type="ref"
data-reference="thm:NB_convergence">3</a>, please look at paper
<a href="#Ng" data-reference-type="ref" data-reference="Ng">[Ng]</a>.

Now we know the parameters of Naive Bayes converge logarithmically to
its optimal values, but this doesn’t directly imply the error of Naive
Bayes also converges with this rate. To intuitively show why the error
also converges, we can first show that the convergence of the parameters
implies that $h_{NB,N}$ is very likely to make the same predictions as
$h_{NB,\infty}$. Recall that $h_{NB}(x)$ makes its predictions according
to
$$\ell_{NB}(x)=\log \frac{\hat{P}(Y=1)\prod_j \hat{P}(x_j|Y=1)}{\hat{P}(Y=0)\prod_j \hat{P}(x_j|Y=0)} > 0$$
For every example for which both $\ell_{NB}(x)$ and
$\ell_{NB,\infty}(x)$ have the same sign, $h_{NB, N}$ and
$h_{NB,\infty}$ will make the same prediction. If $h_{NB,\infty} \gg 0$
or $h_{NB,\infty} \ll 0$, as $\ell_{NB}(x)$ is only a small perturbation
of $\ell_{NB,\infty}(x)$, they will be on the same side of $0$ with high
probability.

With some steps of derivation, we will eventually reach the formal risk
convergence of Naive Bayes

<p id="thm:NB_risk_convergence" class="theorem">
*Define
$G(\tau) = P_{(x,y)\sim \mathbb{P}}\left[ (\ell_{NB,\infty}(x) \in [0,\tau d] \wedge y=1 ) \vee  (\ell_{NB,\infty}(x) \in [-\tau d,0] \wedge y=0 ) \right]$.
Assume that for some fixed $\rho_0 > 0$, we have
$\rho_0 \leq \mathbb{P}(y=1) \leq 1-\rho_0)$, and that
$\rho_0 \leq \mathbb{P}(x_j=1|y=b) \leq 1-\rho_0)$ for all $j$,$b$, then with
high probability,
$$\epsilon(h_{NB,N}) \leq \epsilon(h_{NB, \infty}) + G\left(O\left(\sqrt{\frac{1}{N}\log d}\right)\right)$$</p>

Here, $G$ defines the fraction of points that are very close to the
decision boundary. Intuitively, if we can understand and control the
event $G(\tau)$, then we can obtain a more precise control on the bound
of the error.

<p id="thm:NB_risk_convergence_final" class="theorem">
*Let the conditions of Theorem
<a href="#thm:NB_risk_convergence" data-reference-type="ref"
data-reference="thm:NB_risk_convergence">4</a> hold, and suppose
$G(\tau) \leq \epsilon/2 +F(\tau)$, for some function $F(\tau)$ that
statisfies $F(\tau) \to 0$ as $\tau \to 0$, and some fixed
$\epsilon_0 > 0$. Then for
$\epsilon(h_{NB,N}) \leq \epsilon (h_{NB,\infty}) + \epsilon_0$ to hold
with high probability, it suffices to pick $N=\Omega(\log d)$
</p>

Thus, we can conclude that though the asymptotic error of Naive Bayes is
greater than Logistic Regression, the convergence rate of NB is only
$\Omega(\log d)$, which is faster than $\Omega(d)$ of Logistic
Regression.
