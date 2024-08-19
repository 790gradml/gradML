---
layout: page
title: Introduction
nav_order: 3
---

# Introduction

{: .no_toc}

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

for demo 

## Background
The term “Machine Learning” was coined by MIT alumnus Arthur Samuel[^1]
in 1959. It evolved from many fields including Statistical Learning,
Pattern Recognition and so on. The goal of machine learning is to make
computers “learn” from “data”[^2]. From an end user’s perspective, it is
about understanding your data, make predictions and decisions.
Intellectually, it is a collection of models, methods and algorithms
that have evolved over more than a half-century now.

Historically both disciplines evolved from different perspectives, but
with similar end goals. For example, Machine Learning focused on
“prediction” and “decisions”. It relied on “patterns” or “model” learnt
in the process to achieve it. Computation has played key role in its
evolution. In contrast, Statistics, founded by statisticians such as
Pearson and Fisher, focused on “model learning”. To understand and
explain “why” behind a phenomenon. Probability has played key role in
development of the field. As a concrete example, recall the ideal gas
law $PV = nRT$ for Physics. Historically, machine learning only cared
about ability to predict $P$ by knowing $V$ and $T$, did not matter how;
on the other hand, Statistics did care about the precise form of the
relationship between $P, V$ and $T$, in particular it being linear.
Having said that, in current day and age, both disciplines are getting
closer and closer, day-by-day, and this class is such an amalgamation.

Artificial Intelligence’s stated goal is to *mimic human behavior in an
intelligent manner*, and to do what humans can do but really well, which
includes artificial “creativity” and driving cars, playing games,
responding to consumer questions, etc. Traditionally, the main tools to
achieve these goals are “rules” and “decision trees”. In that sense,
Artificial intelligence seeks to create *muscle* and *mind* of humans,
and *mind* requires learning from data, i.e. Machine Learning. However,
Machine Learning helps learn from data beyond mimicking humans. Having
said that, again the boundaries between AI and ML are getting blurry
day-by-day.

## Course Structure

The course contains four parts:

- Part I. Supervised Learning. Learning from data to
  predict.

- Part II. Unsupervised Learning. Understanding the
  structure within the data.

- Part III. Probabilistic Modeling. Probabilistic view to
  model complex scenarios.

- Part IV. Decision Making. Using data to make decisions.

### Supervised Learning

The goal of supervised learning is to predict *target* using *input* /
*features*, and a model is learned to do so. This can be sufficiently
summarized as
$$\text{\textit{target}} = f ( \text{\textit{features}} )$$ For
classification tasks, the *target* is categorical or takes discrete
values (e.g. hot or cold). For regression tasks, the *target* takes any
real value (e.g. temperature). The model type reflects our *belief*
about the reality and different model leads to different algorithm. The
philosophy of supervised learning is: *future of the past equals future
of the future*.

Examples of classification include: identify handwritten digits, email
spam filtering, detecting malicious network connection based on network
log information or predicting whether a client will default on her/his
credit based on the client’s features. For example, suppose we have
access to a client’s features or attributes in terms of the (credit
card) balance and income. Consider Figure
<a href="#fig:classify" data-reference-type="ref"
data-reference="fig:classify">1</a>. It plots available data with $X$
axis representing (credit card) balance and $Y$ axis representing
income. The color of the point is blue if *no default* and brown if
*default*. Pictorially, the classifier is trying to learn a boundary as
shown in Figure <a href="#fig:classify" data-reference-type="ref"
data-reference="fig:classify">1</a> which separate *no default* from
*default*.

<figure id="fig:classify">
<embed src="./classification.png" style="width:40.0%" />
</figure>

Formally, the data are labeled observations of the following form:
$(x_1,y_1),\ldots,(x_N,y_N)$. The goal is to learn a model that maps
*attribute* (or *feature*) $x$ to *label* (or *target*) $y$ so that
given *attribute* $x$, we can predict corresponding *unknown (discrete)
label* $y$. That is, to learn a function $f$ such that $y = f(x)$ (and
sometimes also what’s the *confidence*).

Various approaches for learning $f$ can be categorized as

- Linear: Logistic regression, Support Vector Machine (SVM), Linear
  Discriminant Analysis (LDA), Perceptron

- Non-linear (parametric): Quadratic Discriminant Analysis (QDA),
  Polynomial, Neural Networks

- Non-parametric: Kernels, Nearest Neighbors

How to find $f$? Among all possible choices of $f$, choose the one that
*fits* the data the best. That is, solve optimization: *empirical risk
minimization (ERM)*:

$$\text{Minimize } \sum_{i=1}^N \mathrm{loss}\left(y_i,f(x_i)\right) \text{ over all possible } f.$$

Stochastic Gradient Descent (SGD) is a method to solve this optimization
problem. This is where Optimization meets Machine Learning.

6.390 (or equivalent undergraduate class) discusses the “How” or
“mechanics” of such approaches. In this class, we expect that you know
the “How” for much of supervised learning and decision making. That is,
more than $60\%$ of this class. So, what will we do in 6.790 (since
$> 60\%$ is already done!)?

To start with, we will learn “Why” behind the “How”. We will utilize
*Probability* as our formal language. We will discuss estimators and
theoretical guarantees, and generalization: does a good model fit on
*historical data* lead to ability to predict *future*? Finally, we will
have 40% of the course discusses unsupervised learning / probabilistic
modeling to understand the structure within the data.

To understand “Why”, effectively we need to “logically deduce” what we
do starting with appropriate goals and axioms. The axioms that are
relevant are that of Probability. In particular, to reason about what we
do in Machine Learning, we will utilize the language of probability. And
probability is entirely based on the three key axioms. Formally, there
is a probability space $\Omega$, events $\mathcal{F}$ in it, and a
probability function $\mathbb{P}:\mathcal{F}\to [0,1]$.

- Axiom 1. $\mathbb{P}(A)\geq 0$, for all $A\in\mathcal{F}$.

- Axiom 2. $\mathbb{P}(\Omega)=1$.

- Axiom 3.
  $\mathbb{P}(\cup_{i=1}^\infty E_i)=\sum_{i=1}^\infty \mathbb{P}(E_i)$,
  if $E_i\cap E_j=\varnothing$, for all $i\neq j$.

The above exercise is a simple example of logical deduction starting
from the axioms of probability. In a sense, this is what we will do to
explain “why”.

Before proceeding further, it is important to wonder – “Is it possible
to have a different set of probability axioms?” This is a question hotly
debated in the first half of last century. At the end of the day, *All
roads lead to Rome*: All sorts of reasonable hypothesis about beliefs /
decision making lead to axioms of probability[^3].

In the language of probability, both attributes $X$ and labels $Y$ are
random variables. Especially, $Y$ is discrete-valued random variable.
The conditional distribution $\mathbb{P}(Y|X)$ is of interest. Suppose
labels take value $1$ (e.g. default) or $-1$ (e.g. no default), given
attribute $X=x$. An ideal classier, also known as *Bayes classifier*,
which in the context of binary classification, predicts

$$
\hat{Y}(x)= \begin{cases}1 & \text { if } \mathbb{P}(Y=1 \mid X=x) \geq 1 / 2 \\ 0 & \text { otherwise }\end{cases}
$$

The performance metric of interest is mis-classification
probability, i.e. $\mathbb{P}(\hat{Y}(X) \neq Y)$.

Probabilistic view will help us understand how to choose the loss
function and how well our model *generalizes*. In terms of
generalization and overfitting, you should trust your data, but only so
much. Consider the following example: We have observations $(x_i,y_i)$,
$i=1,\ldots,n$. Here attributes ${x}_i$ are points distributed uniformly
in the unit square. The label is generated according to the following
rule: As sketched in the figure below, $y_i=0$ when the corresponding
$\mathbf{x}_i$ lies in the shaded square and $y_i=1$ otherwise. The area
of the shaded square is $1/2$.

<figure>
<embed src="./square.png" style="width:22.0%" />
</figure>

Pretend we do not know the true label rule and would like to to find a
model to approximate it based on the observations. The function fit,
$$f({x})=\begin{cases}
    y_i, & \text{if }{x}={x}_i, \\
    0, & \text{otherwise},
    \end{cases}$$ which assigns every observed points to the correct
label $y_i$ and assign all unseen points to $0$, is a perfect fit for
the observation. However, since the possibility we encounter the same
points in the set $\{({x}_i,y_i), i=1,\ldots,n\}$ in the future is zero,
we will most certainly assign all future points to $0$ and this function
is simply as bad as “random” function! This is overfitting.

In order to prevent overfitting, empirically, we use *cross-validation*
– split data into three parts: *train*, (*validate*) and *test*, or/and
*$K$-fold* cross-validation. To explain why this the right thing to do,
we shall discuss the notion of *generalization* that utilizes the view
that data is generated per an unknown underlying probability
distribution. Methodically, we shall use *regularization* and again
probabilistic formalism will help explain why (or why not) it works
well. Probabilistic view, again will come to our rescue to explain the
*implicit* regularization that is implemented by modern methods (e.g.
‘dropout’) of neural networks.

Some examples of regression include predict wage given age, year, and
education level. Formally, the data are labeled observations of the
following form: $(x_1,y_1),\ldots,(x_N,y_N)$. The goal is to learn a
model that maps *attribute* (or *feature*) $x$ to *label* (or *target*)
$y$ so that given *attribute* $x$, we can predict corresponding *unknown
(continuous) label* $y$. That is, to learn a function $f$ such that
$y = f(x)$ (and sometimes also what is the *confidence interval*).

In the language of probability, both attributes $X$ and labels $Y$ are
random variables. Now, $Y$ is continuous-valued random variable. The
conditional distribution $\mathbb{P}(Y|X)$ is of interest. Given
attribute $X=x$, we estimate $\hat{Y}(x)$ to minimize estimation error.
One the most common estimation error is
$\mathbb{E}\left[(Y-\hat{Y}(x))^2|X=x\right]$, which is minimized by
$\hat{Y}(x)=\mathbb{E}\left[Y|X=x\right]$. Finally, we should determine
*predictive* distribution. $\mathbb{E}\left[Y|X=x\right]$ is unknown.
The model fit for regression means to find the best fit for
$f(x)\approx \mathbb{E}\left[Y|X=x\right]$ using observed data.

### Unsupervised Learning

In unsupervised learning, there is no *target*. Only *input* /
*features* are given. The goal is to learn the data distribution. In
this course, we are going to cover topics such as dimensionality
reduction, matrix estimation, clustering and mixture distribution, and
feature extraction (topic model and deep generative model) from
unstructured data such as text, audio or image, or for complexity
reduction. Examples of unsupervised learning: Finding the principal
component of DNA data (dimensionality reduction) , movie recommendation
(matrix estimation), analyzing topics in documents (feature extraction:
topic model), generating fake faces of celebrities (feature extraction:
deep generative model).

### Probabilistic Modeling

Two important topics in probabilistic modeling are:
-  incorporating prior
knowledge from Bayesian perspective, and 
- sampling from distribution when
probabilistic model is complex.

Most of the key tasks in machine learning are inference tasks. For
example, in prediction we need to infer $\mathbb{P}(Y|X)$. In model
learning, we need to infer $\mathbb{P}(\text{parameters}|\text{data})$.

The Bayes’ rule states that

$$\underset{\text{posterior}}{\mathbb{P}(\text{parameters}|\text{data})}\propto \underset{\text{likelihood}}{\mathbb{P}(\text{data}|\text{parameters})}\times \underset{\text{prior}}{\mathbb{P}(\text{parameters})}$$

The key question is how to select *prior*? This is the *prior* knowledge
of the world. One of the classical priors is Gaussian distribution,
which for example, leads to ridge regularization in regression.

A probability distribution can be complex. It may have succinct
representation but no closed-form formula, and hence difficult to
evaluate. For example, we may know
$$\mathbb{P}(X=x)\propto \exp(f(x))=\frac{1}{Z}\exp(f(x)),$$ where
$$Z=\int  \exp(f(x))dx.$$

This integration can be very hard to evaluate
for a general $f(x)$. The key algorithm to evaluate on such complex
distributions is Markov Chain Monte Carlo (MCMC)[^4] It has specific
forms such as Gibbs sampling and Metropolis-Hastings. MCMC works for
generic form of distribution.

### Decision Making

In data driven decision making (in presence of uncertainty), we need to
learn the model of uncertainty, given observations. The goal is to make
“optimal” decision with respect to a long-term objective. The decision
vs information *timescale* are critically important. The following
diagram summarizes the framework of decision making,

<figure>
<embed src="./dm.png" style="width:50.0%" />
</figure>

The two key *timescales* are state or environment dynamics, and
information dynamics. Depending on the two timescales, there are methods
/ approaches including optimizing given model of uncertainty, Markov
decision process, and reinforcement learning.

|                                       | State Dynamics                | Information Dynamics                 |
|:--------------------------------------|:------------------------------|:-------------------------------------|
| Optimizing Given Model of Uncertainty | No change (or extremely slow) | Lots of historical information       |
| Markov Decision Process               | High                          | Lots of historical information       |
| Reinforcement Learning                | High                          | Minimal information, learn as you go |

One fundamental challenge in reinforcement learning is *explore vs
exploit*: we need to make decisions based on incomplete information about the world; in turn the decision we make affects future data we get to see. so should we stick to a decision that seems best for the moment, or sacrifice near-term gain to get more information for future long-term decision making. 

Another central question in RL is the **credit assignment problem**. (TO be expanded.)

Reinforcement learning has found success in various applications such as robotics, automated game player, ads placement. And notably, in the recent success of large-language-models (LLMs) such as [ChatGPT](https://chat.openai.com). 
## And then, What Is Not Covered, But Of Interest

We may not be able to cover the following interesting topics in machine
learning:

- Active Learning, actively obtain data as each data point is expensive.

- Transfer Learning, transfer data collected for one task to other
  learning task.

- Semi-supervised Learning, supervised setting with (additional)
  unsupervised data.

- Causal inference, Hypothesis testing, ...

But hopefully, things you’ll learn this in course will provide
systematic foundations to approach these topics.

[^1]: See <https://g.co/kgs/Lj3v3k> to read more about Arthur Samuel.

[^2]: What is learning? Some food for thought: <https://goo.gl/5R1m4S>.

[^3]: A good set of readings include , and

[^4]: MCMC is one of the top 10 algorithms of all time . Other algorithms include quicksort and fast Fourier transform.
