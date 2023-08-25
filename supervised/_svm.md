---
layout: about
title: SVM
parent: Supervised Learning
nav_order: -2
---

# Overview

In the earlier lecture, we introduced formal setup and notations; in
this lecture, we'll make do a concrete utilization of it. We will start
by continuing our discussion of the ERM framework and talk about how
this reduces to the finite-sum optimization and how the (stochastic)
gradient descent can be used to solve for it. We then will focus on the
class of linear classifiers, specifically the support vector machines
(SVMs), which are very widely used. They provide a geometrically
motivated solution for classification. We will discuss the SVM margin
maximization setup, and associated geometric interpretation. From there,
we will show how linear classifiers are in fact capable of nonlinear
classification, if coupled with an appropriate nonlinear feature
mapping; and we will then motivate the so-called kernel trick that
allows us to not explicitly carry out such (often-times very complex)
nonlinear mapping for the task. In this notes, additional (bonus)
material is included that is beyond what is covered in the class; we
will mark the associated sections with $\star$.

# ERM and Optimization

Recall that we defined the risk as the expected loss of
$h \in \mathcal{H}$ with respect to the data distribution $\mathbb{P}$
over $\mathcal{X} \times \mathcal{Y}$
$$L(h) :=\mathbb{E}[\ell(h, X, Y)].$$ The goal is to find $h$ that
minimizes $L(h)$ over a given hypothesis class ${\cal H}$. The ERM
framework suggests using the actual data as proxy to evaluate risk and
minimize it since we do not have access to $\mathbb{P}$. Specifically,
empirical risk
$$L_{S}(h) :=\frac{1}{N} \sum_{i=1}^{N} \ell\left(h, x_{i}, y_{i}\right).$$
Therefore, the optimization problem takes form $$\begin{aligned}
{\sf minimize}\quad L_S(h) & \quad {\sf over} ~h \in {\cal H},
\end{aligned}$$ or equivalently, if each hypothesis is parameterized by
parameter $\theta$, the above takes form of $$\begin{aligned}
{\sf minimize}\quad \frac1N\sum_i f_i(\theta) & \quad {\sf over}\quad \theta,
\end{aligned}$$ where $f_i(\theta) = \left(h, x_{i}, y_{i}\right)$ where
$\theta$ corresponds to hypothesis $h$. This is a *finite-sum
optimization problem* and variety of model learning problems fall within
this framework. For example, the classical least square problem states
that find parameter $\theta$, so that the inner product of $\theta$ with
feature approximations the labels well. That is, $$\begin{aligned}
\min _{\theta} f(\theta) &=\frac{1}{2}\|X \theta-y\|^{2} ~
=\frac{1}{2} \sum_{i=1}^{N} \underbrace{\left(x_{i}^{T} \theta-y_{i}\right)^{2} }_{f_{i}(\theta) }.
\end{aligned}$$ The neural network training usually involves solving a
similar function.

The primary question of interest: how does one solve such an
optimization problem? Gradient descent (GD) is an excellent method for
solving such problems. Precisely, it is an iterative algorithm starting
with some initialization $\theta^0$, and for $k \geq 1$, updating it as
$$\theta_{k+1}=\theta_{k}-\eta_{k} \frac{1}{N} \sum_{i=1}^{N} \nabla f_{i}\left(\theta_{k}\right).$$
As can be seen, this requires computation that scales linearly with
number of data points $N$ for each iteration. To overcome this challenge
of scaling with data, stochastic gradient descent (SGD) has been often
utilizes which, instead of using all $N$ points for computing gradient,
at each iteration, chooses a smaller number of data points at random to
compute the gradient. This has made SGD the algorithm of choice for
large scale machine learning applications. Figure
[1](#GDSGD){reference-type="ref" reference="GDSGD"} demonstrates how SGD
is usually more "jerky" than GD since SGD uses the gradient of one
sample as the proxy for the average gradient of the entire set.

# Linear models

Let's focus on hypothesis class ${\cal H}$ that contains only 'linear'
functions. With slight abuse of notation[^1], such hypothesis class can
be represented by parameters
$\{(w, w_0): w \in \in \mathbb{R}^{d}, w_{0} \in \mathbb{R}\}$. Given
parameter $(w, w_0)$, the associated classifier that maps
$x \in \mathbb{R}^d$ to $\{-1, 1\}$, is given by
$$h\left(x ; w, w_{0}\right)=\operatorname{sign}\left(w^{T} x+w_{0}\right)=\left\{\begin{array}{ll}{+1,} & {w^{T} x+w_{0}>0}, \\ {-1,} & {w^{T} x+w_{0} \leq 0},\end{array}\right.$$
In effect, such a classifier always satisfies
$h(x; w, w_0) \left(w^{T} x_{i}+w_{0}\right) \geq 0$. Therefore, the
classification error can be equivalently written as
$$y_i \neq h(x_i; w, w_0) ~\Leftrightarrow~ y_i \left(w^{T} x_{i}+w_{0}\right) < 0.$$
Therefore, the associated ERM takes form $$\begin{aligned}
{\sf minimize}\quad \frac1N \sum_{i} \operatorname{sign}(- y_i \left(w^{T} x_{i}+w_{0}\right)).
\end{aligned}$$ Or, more abstractly $$\begin{aligned}
{\sf minimize}\quad \frac1N \sum_{i} \ell( y_i \left(w^{T} x_{i}+w_{0}\right)),
\end{aligned}$$ where $\ell(\cdot)$ is the loss function. Depending upon
the choice of $\ell$, beyond the $\operatorname{sign}(- \cdot)$, we get
different classifiers: $\ell_{\log }(z) :=\log \left(1+e^{-z}\right)$ is
called the logistic loss and we get logistic regression;
$\ell_{h}(z) :=\max (0,1-z)$ is called the hinge loss and we get
(almost) SVM (except for the most important part (for reference, it's
the margin that's missing, which we'll cover in detail later).

## Logistic Regression $\star$

Logistic regression is still very widely used. For example, currently it
is a crucial "sub-routine" utilized within the complex zoo of
algorithmic architecture that enables online advertisement. Here we
explain the precise algorithm and its derivation. Logistic regression
start with the goal to obtain Bayes classifier for linear hypothesis
class, and use the empirical estimate of the true distribution as a
proxy. In other words, we would like to use linear model to estimate
$\hat{\eta}(x)=\mathbb{E}[Y | X=x]$, and find the classifier:
$h(x)={[\![\hat{\eta}(x) \geq \frac{1}{2}]\!]}$. That is, we need to
find a good approximation to step function
${[\![\eta(x) \geq \frac 12]\!]}$.

The difficulty with the 'obvious' use of linear model, i.e.
$\hat{\eta}(x)=f(x)=w^{T} x+w_{0}$, is that it is a poor approximation
to a step function and it can lead the predicted value to be outside
$[0,1]$, i.e. it wont be probability. We need to use a smoother
approximation that contains value within $[0,1]$. To that end, logistic
regression makes choice of using sigmoid function:
$$\hat{\eta}(x)=f(x)=\sigma\left(w_{1} x+w_{0}\right)=\frac{e^{w_{1} x+w_{0}}}{1+e^{w_{1} x+w_{0}}},$$
because it maps all of $\mathbb{R}$ into the interval ${[0,1]}$, and
moreover this map is bijective (so it offers a "change of coordinates").
The resulting 'decision boundary' or 'classification region boundary' is
given by
$$\begin{array}{c}{\frac{1}{1+e^{-w^{T} x}}=0.5} {\Longrightarrow e^{-w^{T} x}=1} {\Longrightarrow w^{T} x=0}.\end{array}$$
That is, logistic regression is indeed linear classifier. There are some
important reasons behind the popularity of LR:

- *Easy to fit*: Ideal example for industrial scale data with scalable
    parameter estimation methods.

- *Easy to interpret*: Consider log odds defined as
    $$\mathcal{L O}(x) :=\log \frac{p(y=1 | x)}{p(y=0 | x)}=w^{T} x,$$
    Let's use an example to see how we could interpret this: suppose
    data has two features, number of hours you study and time in days it
    took you to do homework. The goal is to predict the probability you
    will get an A. Suppose, LR estimates the parameters to be
    ${w=(1.3,-1.1)}$. This means that for every extra hour you study,
    your chance of getting an A increases by a factor of $\exp(1.3)$.

- *Easy to extend*: to multiclass (softmax), nonlinear features,
    kernels, or even Neural networks.

Now the question is how do we learn the parameters associated with the
logistic regression. We will utilize the maximum likelihood approach to
do so. The maximum likelihood estimator (MLE) suggests finding
parameters so that the likelihood of observation is maximized.
Precisely, the likelihood of data given parameter $w$ is given by
$$\begin{aligned}
\ell(w) &  ~=~\prod_{i : y_{i}=1} p\left(x_{i}\right) \prod_{j : y_{j}=0}\left(1-p\left(x_{j}\right)\right)
\end{aligned}$$ Therefore, the negative of log-likelihood is
$$\begin{aligned}
\mathcal{L}(w) & = - \log \ell(w)  ~=-\sum_{i=1}^{N} \log \left[\sigma\left(w^{T} x_{i}\right)^{y_{i}}\left(1-\sigma\left(w^{T} x_{i}\right)\right)^{1-y_{i}}\right] \nonumber \\
&=-\sum_{i=1}^{N}\left[y_{i} \log \sigma\left(w^{T} x_{i}\right)+\left(1-y_{i}\right) \log \left(1-\sigma\left(w^{T} x_{i}\right)\right)\right].\label{eq.crossent}
\end{aligned}$$ Notice the compact and convenient notation of using
$\sigma(\cdot)^y (1-\sigma(\cdot))^{1-y}$ to represent the likelihood of
data under the model of logistic regression. The above function
[\[eq.crossent\]](#eq.crossent){reference-type="eqref"
reference="eq.crossent"} is called the *cross-entropy* error function
and is utilized beyond the logistic regression. If we are using labels
$\{-1,1\},$ then we have $p(y=1 | x)=\frac{1}{1+e^{-w^{T} x}}$,
$p(y=-1 | x)=\frac{1}{1+e^{+w^{T} x}}$ and the cross-entropy becomes:
$$\begin{aligned}
\mathcal{L}(w) & =\sum_{i=1}^{N} \log \left(1+\exp \left(-y_{i} w^{T} x_{i}\right)\right). \label{eq.crossent.1}
\end{aligned}$$ And the Maximum Likelihood Estimation (MLE) becomes the
minimization problem $$\begin{aligned}
{\sf minimize} ~~L(w)~ & \quad {\sf over} ~ w \in \mathbb{R}^d.
\end{aligned}$$ Given the form
[\[eq.crossent.1\]](#eq.crossent.1){reference-type="eqref"
reference="eq.crossent.1"}, it can be checked that the above
optimization problem is convex minimization. Therefore, SGD can be shown
to converge to an optimum at the rate $O(1 / \sqrt{T})$ where $T$ is the
number of updates [@SGD:REF]. Formally, SGD can be described in
pseudo-code as follows.

Initialize weights $w$. Set ${t = 0}$ Pick $i$ in $\{1, \ldots, N\}$
Obtain subgradient of $\ell_{\log }\left(y_{i} w^{T} x\right)$ Update
$w^{t+1}=w^{t}-\eta_{t} g_{t}$ **return** $w$

An important remark is needed. Though the minimization problem is
'nice', the unconstrained minimizer obtained by setting gradient to $0$,
i.e. $\nabla_w \mathcal{L}(w)= 0$ is achieved only when
$\|w\| \to \infty$! Therefore, it is essential to add 'constraint' to
the optimization problem to make it well behaved. This can be achieved
in many different ways. One reasonable option is to add $\|w\|_2^2$ as a
penalty term corresponding to regularization. More generally, adding
penalty term $\|w\|_q^p$ can lead to different structural assumption on
the model of interest. For example, $q = p = 1$ will encourage sparsity
in the model.

Thus far, we have discussed logistic regression (and more generally
classification) for the setting of binary labels. However, it naturally
extends to setting beyond binary. In effect, for each potential label
value $k \in \{1,\dots, L\}$ amongst $L \geq 2$ possible labels, one
models $$\begin{aligned}
\mathbb{P}(Y = k | X = x) & \propto \exp\Big(w(k)^T x\Big),
\end{aligned}$$ where $w(k) \in \mathbb{R}^d$ is parameter associated
with label taking value $k \in \{1,\dots, L\}$. Again, very similar
approach as in the case of binary set up extends for this scenario to be
able to learn parameters $w(k), k \in \{1,\dots, L\}$ as well as it
provide interpretation. An interested reader is referred to textbook for
further details.

## Support Vector Machine (SVM)

The Support Vector Machine (SVM) is another type of linear
classification method. We discuss this method in detail here.

### Learning hyperplanes

We would like to develop intuition in terms of linear classification
being the question of learning hyperplane in the feature space. To that
end, consider the data shown in
Figure [2](#fig:one){reference-type="ref" reference="fig:one"}. It shows
each data point's representation in two dimensional feature space and
the color associated with each point is one of the two binary labels
associated with. The goal of the linear classifier for this binary
classification problem is to identify the separating hyperplane between
the points of different colors, if feasible. The figure shows different
such candidates. The question is, which is more preferable and why?

Most people choose to pick hyperplanes that separate the data as
"widely" as possible, guided by the intuition that this leads to the
sharpest separation and perhaps the least chance of confusing positive
points with negative points on new data. In the rest of this section, we
make this viewpoint mathematically more precise.

### Separating hyperplanes

A hyperplane in a $\mathbb{R}^d$ is characterized by
$(w, w_0) \in \mathbb{R}^d \times \mathbb{R}$ and formally defined by
set of points $$\label{eq:1}
  \mathcal{H}(w, w_0) = \ifmmode{\left\{ x \in \mathbb{R}^d \mid w^Tx + w_0 =0 \right\}}\else\mbox{$\left\{ x \in \mathbb{R}^d \mid w^Tx + w_0 =0 \right\}$}\fi,$$
where $w$ denotes a vector normal to the hyperplane and $w_0$ the
offset, which is called *bias*. If we scale $w$ and $w_0$ by a constant
$\delta$ then the hyperplane remains unchanged. While trying to fit a
hyperplane to training data, we thus make an *arbitrary* scaling choice
and consider *canonical hyperplanes*, such that for our training data
$\ifmmode{\left\{ x^{(1)},\ldots,x^{(n)} \right\}}\else\mbox{$\left\{ x^{(1)},\ldots,x^{(n)} \right\}$}\fi$
we have $$\label{eq:2}
  \min_{1\le i \le n}\quad |w^Tx^{(i)}+w_0 |=1.$$ With this
normalization one sees that the point closest to the hyperplane
$\mathcal{H}(w, w_0)$ is at a distance of $1/\left\| {w} \right\|$. To
see how we obtained this value, let us consider the following simple
geometric question.

**Question.** What is the distance of a point $x \in \mathbb{R}^d$ from
$\mathcal{H}(w, w_0)$?\
**Answer.** Recall from Homework0 that if $\bar{x}$ is the projection of
a point $x$ onto a hyperplane, then we can decompose $x$ as
$$\label{eq:3}
  x = \bar{x} + t \frac{w}{\left\| {w} \right\|},$$ that is, into a
component on the hyperplane and a component normal to the hyperplane.
In [\[eq:3\]](#eq:3){reference-type="eqref" reference="eq:3"}, $t$
denotes the signed distance to the hyperplane, i.e., $|t|$ is the
distance of $x$ to the hyperplane. Since
$\bar{x} \in \mathcal{H}(w, w_0)$, it follows that $w^T\bar{x}+w_0 =0$.
Thus, from [\[eq:3\]](#eq:3){reference-type="eqref" reference="eq:3"} we
obtain
$$w^Tx+w_0 -t\frac{w^Tw}{\left\| {w} \right\|} = w^T\bar{x}+w_0  = 0,$$
which yields $$t = \frac{w^Tx+w_0 }{\left\| {w} \right\|},$$ so that the
distance of a point $x$ to $\mathcal{H}(w, w_0)$ is
$|w^Tx+w_0 |/\left\| {w} \right\|$. But recall that we are considering
*canonical hyperplanes*, i.e. for all data points $x$,
$|w^Tx+w_0| \geq 1$. And there is one point with $|w^Tx+w_0| = 1$. That
is, the closest point must have distance equal to
$1/\left\| {w} \right\|$. This completes the answer to the question.

![Separating hyperplane with large margin. The normal direction is given
by $w/\left\| {w} \right\|$ and the red points lie on the positive side
of the hyperplane.](./svms3.png){#fig:sephyp width=".5\\linewidth"}

Closely related to the signed distance $t$ above is a quantity called
the *geometric margin*: for a given labeled data point $(x, y)$, with
$y$ being the class label ($+1$ or $-1$) of the point $x$, consider
$$\label{eq:4}
  yt = y \frac{w^Tx+w_0 }{\left\| {w} \right\|}.$$ As can be seen from
Fig. [3](#fig:sephyp){reference-type="ref" reference="fig:sephyp"}, if
data is separable using the hyperplane, then $w^Tx+w_0 > 0$ for the
positive points and $w^Tx+w_0  < 0$ for the negative ones. Thus, the
geometric margin $yt > 0$ (for points not on the separating hyperplane).
Since on training data we impose the
normalization [\[eq:2\]](#eq:2){reference-type="eqref"
reference="eq:2"}, we see that the smallest geometric margin over
training data is $1/\left\| {w} \right\|$. We will mostly omit the word
"geometric" and just call it the margin. These observations can be
summarized as follows:

- For a correctly classified point the margin is simply the distance
    to the hyperplane.

- Multiplying by $y$ ensures that the margin is always positive
    whenever its corresponding point $x$ is correctly classified.

- The canonical condition [\[eq:2\]](#eq:2){reference-type="eqref"
    reference="eq:2"} implies a margin $1/\left\| {w} \right\|$.

- We wish to make the margin **as large as possible** (this
    corresponds to the intuition expressed for selecting hyperplanes in
    Fig. [2](#fig:one){reference-type="ref" reference="fig:one"}).

- The classifier that we learn is:
    $h(x) = \mathop{\mathrm{sgn}}(w^Tx+w_0 )$.

### Maximizing Margin: An Intuition

Here, we provide intuitive explanation of why maximizing margin is a
good idea. Specifically, assuming that the training and test data points
have the same underlying distribution. Then, it is reasonable to assume
that most of the test points (except for some outliers) might lie close
to at least one of the training points.

![Robustness of margin](./svms4.png){#fig:margins
width="0.3\\linewidth"}

Suppose (for the sake of illustration) that the test data points are
generated by adding bounded noise to the training data. Thus,
$(x, y) \to (x+\delta x, y)$, where $\left\| {\delta x} \right\| \le r$
for some $r > 0$. Fig. [4](#fig:margins){reference-type="ref"
reference="fig:margins"} illustrates that if we manage to attain a
margin of $\rho > r$ on training data, then we will correctly classify
**all** the test data points, since the training data are at a distance
$\ge \rho$ to the separating hyperplane. This idea is formalized and
studied in much greater depth in statistical learning theory, where
ultimately a precise mathematical statement of the following form can be
proved [@XYZ]: The probability that a test data point is misclassified
is bounded from above by
$$\text{margin error} + \mathcal{O}\left(\frac{1}{\text{margin}}\right),$$
where *margin error* corresponds to the fraction of training examples
with margin smaller than $1/\left\| {w} \right\|$, i.e. the fraction of
training data points that *violated* the margin condition. In other
words, we have to make a bias-complexity tradeoff. We can keep the
margin error small by shrinking the margin (overfitting), but that then
drives up the $\mathcal{O}(\cdot)$ term; while making the margin large
shrinks the second term but increases the chance of margin error
(underfitting to training data). Such bias-complexity (aka
bias-variance) tradeoffs are a recurrent theme in machine learning, and
they should be explored when trying to improve or refine an existing
model. In summary: *Keep margin error (training error) as small as
possible while making the margin as large as possible.* The next section
casts this desideratum into an optimization problem.

### Margin Maximization: An Optimization Perspective

Based on our description above, we now derive an optimization problem.
Assume that we have at least one positive and one negatively labeled
data point. Our aim is to find a decision function
$f_{w,w_0 }(x) = \mathop{\mathrm{sgn}}(w^Tx+w_0 )$ such that
$f_{w,w_0 }(x^{(i)})=y^{(i)}$. Assuming that the training data are
linearly separable (we will deal with the inseparable case afterwards),
our canonical hyperplane
requirement [\[eq:2\]](#eq:2){reference-type="eqref" reference="eq:2"}
implies that $$y^{(i)}(w^Tx^{(i)}+w_0 )\ge 1,\qquad 1\le i \le n.$$
Thus, the task of maximizing the margin while separating the data can be
written as $$\begin{aligned}
{2}
%   \max_{w,w_0 }\quad &\frac{1}{\norm{w}}\quad\text{s.t.}\quad &\min_{1\le i \le n} |w^T\ex{x}{i}+w_0 |=1.
% \end{alignat*}
% Trying to satisfy
  \max_{w,w_0 }\quad &\frac{1}{\left\| {w} \right\|}\qquad\text{s.t.}\quad \min_{1\le i \le n} y^{(i)}(w^Tx^{(i)}+w_0 )=1.
\end{aligned}$$ Let us rewrite this problem as a convex optimization
problem: $$\label{eq:5}
  \begin{split}
    \min_{w,w_0 }\quad &\tfrac12\left\| {w} \right\|^2\\
    &\text{s.t.}\quad y^{(i)}(w^Tx^{(i)}+w_0 ) \ge 1,\quad 1 \le i \le n.
  \end{split}$$ Problem [\[eq:5\]](#eq:5){reference-type="eqref"
reference="eq:5"} is called the **Hard-SVM** (the hard margin SVM)  in
its primal formulation. This problem is convex, with a strictly convex
cost function and linear constraints. If the constraints are feasible,
then the problem has strong duality, whereby the KKT conditions[^2] are
necessary and sufficient. Let us analyze the optimality conditions of
problem [\[eq:5\]](#eq:5){reference-type="eqref" reference="eq:5"}.
Consider the Lagrangian $$\label{eq:6}
  L(w,w_0 ,\alpha) := \tfrac12\left\| {w} \right\|^2 - \sum_i\alpha_i[y^{(i)}(w^Tx^{(i)}+w_0 )-1].$$
We have the following KKT conditions for candidate optimal points
$(w,w_0 ,\alpha)$: $$\begin{aligned}
  \frac{\partial L}{\partial w} = 0,\quad \frac{\partial L}{\partial w_0 } =0&\quad\text{(stationarity)}\\
  \alpha_i[y^{(i)}(w^Tx^{(i)}+w_0 )-1] = 0\quad\forall i &\quad\text{(complementary slackness)}\\
  y_{i}\left(w^{T} x_{i}+w_{0}\right) \geq 1, \forall i &\quad\text{(primal feasibility)}\\
  \alpha_i \ge 0 \quad\forall i&\quad\text{(dual feasibility)}.
\end{aligned}$$ The stationarity condition simplifies to yield
$$w = \sum_i \alpha_i y^{(i)}x^{(i)},\quad \sum_i \alpha_iy^{(i)}=0,$$
while from complementary slackness (and dual feasibility) we see that
when the Lagrange multipliers $\alpha_i > 0$, the corresponding primal
constraint $y^{(i)}(w^Tx^{(i)}+w_0 )=1$ is tight. These training data
points lie closest to the separating hyperplane. In fact, if we recall
the convex hull view of Fig. [2](#fig:one){reference-type="ref"
reference="fig:one"}, it can be shown that these points lie on
supporting hyperplanes of the respective convex hulls. It is for this
reason that these points are called **support vectors**.

![Support vectors, marked in yellow, define the extent of the
margin.](./svms5.png){#fig:svecs width="0.35\\linewidth"}

It is worth noticing that all the remaining training examples are
irrelevant, as their constraints are satisfied automatically. Thus, if
we were to discard them, the obtained separating hyperplane would be the
same. Of course, we do not know *a priori* which points will be support
vectors, so we cannot discard non support vectors in advance (although
this idea plays a role in the design of certain "active-set" methods for
the more general SVM problem that we will soon consider). Moreover,
observe that only the support vectors contribute to the classifier that
we learn, because we have $w = \sum_i \alpha_iy^{(i)}x^{(i)}$, whereby
only the training data points that become support vectors contribute to
the classifier (which predicts $\mathop{\mathrm{sgn}}(w^Tx+w_0 )$ for a
test pattern $x$). This "sparse" nature of SVMs has also contributed to
their popularity (and potential computational speed at test time).

**Dual problem.$\star$** Corresponding to the primal
problem [\[eq:5\]](#eq:5){reference-type="eqref" reference="eq:5"}, upon
considering the Lagragian [\[eq:6\]](#eq:6){reference-type="eqref"
reference="eq:6"} we arrive at the dual problem $$\begin{aligned}
 \label{eq:7}
    {\sf maximize}_{\alpha\in \mathbb{R}^n} & \left[g(\alpha) := \min_{w,w_0} L(w,w_0,\alpha) \right] ~= -\frac{1}{2}\left\| {\sum\nolimits_i\alpha_iy^{(i)}x^{(i)}} \right\|^2 + \sum\nolimits_i \alpha_i\\
     {\sf such~~ that} & \quad \alpha_i \ge 0,\ \ 1 \le i \le n \nonumber \\
    & \quad \sum\nolimits_i y^{(i)}\alpha_i = 0. \nonumber
\end{aligned}$$

### Alternative derivation with convex hulls$^\star$

We briefly digress to mention an alternative geometric way to arrive at
the Hard-SVM. The material of this section is optional and can be
skipped at first reading.

We follow the convex-hull view suggested by
Fig. [2](#fig:one){reference-type="ref" reference="fig:one"}. It can be
seen that the task of maximizing the margin amounts to finding the
closest that the hulls of the positive and negative examples come
towards each other. We leave it to the reader to verify that the
following optimization formulation achieves this aim. $$\label{eq:8}
  \begin{split}
    {\sf minimize}_{c}\quad &\left\| {\sum\nolimits_{y^{(i)}=+1}c_ix^{(i)} -\sum\nolimits_{y^{(i)}=-1}c_ix^{(i)}} \right\|^2\\
    {\sf such~that}~\ &\sum_{y^{(i)}=+1}c_i=1,\quad \sum_{y^{(i)}=-1}c_i=1, \quad c_i \ge 0,\ \forall i.
  \end{split}$$ We need to recover the scale of $w$ from this and to
also suitably adjust $w_0$. If you are interested in learning more about
this view (including its extension to the linearly inseparable case),
please refer to the paper "Duality and Geometry in SVM Classifiers" by
Bennett and Bredensteiner.

## C-SVMs

A careful reader may notice that in the discussion above, we assumed
that the primal problem is feasible. However, as is usually the case,
there may be no hyperplane that separates the training data points. In
this case, problem [\[eq:5\]](#eq:5){reference-type="ref"
reference="eq:5"} is infeasible and has no solution. We get around this
difficulty by relaxing the margin constraints by introducing slack
variables. In particular, we consider the following **Soft-SVM**
problem: $$\begin{aligned}
 \label{eq:9}
    {\sf minimize}_{w,w_0,\xi} & \quad \frac{1}{2}\left\| {w} \right\|^2 + C\sum\nolimits_i\xi_i\\
    {\sf such~that~}& \quad y^{(i)}(w^Tx^{(i)}+w_0) \ge 1 - \xi_i,\quad 1 \le i \le n \nonumber \\
    & \xi_i \ge 0, 1 \le i \le n. \nonumber
\end{aligned}$$ Observe that by making $\xi_i$ large enough the
constraints can always be met. Indeed, if we set $\xi_i=1$ and $w=0$ and
$w_0=0$, the constraints are always satisfied and cost is also reduced.
To avoid such a trivial solution we penalize the amount of "slack"
introduced into the constraints. This penalty is achieved by adding the
term $C\sum_i \xi_i$ to the objective function, where $C > 0$ is a
hyper-parameter.

![Linearly inseparable data. One of the green points lies on the "wrong"
side of the separator, so we have to tolerate training errors while
seeking a large margin hyperplane.](./svms6.png){#fig:notsep
width="0.45\\linewidth"}

As can be seen from [\[eq:9\]](#eq:9){reference-type="eqref"
reference="eq:9"}, whenever $\xi_i=0$, the margin constraint is met, and
the corresponding training data point is *not* a margin error. All the
nonzero slacks correspond to margin errors. Thus,
formulation [\[eq:9\]](#eq:9){reference-type="eqref" reference="eq:9"}
makes the tradeoff between the margin width and margin errors apparent,
and the amount by which we increase or decrease the importance of
training errors is controlled by choosing $C$. If the slack cost
$\sum_i \xi_i$ is significantly larger than the fraction of margin
errors, it is likely that the hyperplane obtained by
solving [\[eq:9\]](#eq:9){reference-type="eqref" reference="eq:9"} does
not generalize well.

The above described Soft-SVM problem can be rewritten in the usual
"Regularized" ERM format as follows: $$\label{eq:9.1}
  \min_{w,w_0}\ L_S(w,w_0) := \tfrac12\|w\|^2 + \frac{C'}{N}\sum_{i=1}^N \max(0,1-y_i(w^Tx_i+w_0)).$$
The reader should try to obtain this formulation
from [\[eq:9\]](#eq:9){reference-type="eqref" reference="eq:9"}.

::: center
:::

**Dual problem.$\star$** Forming the Lagrangian
of [\[eq:9\]](#eq:9){reference-type="eqref" reference="eq:9"} as before,
and simplifying we obtain the following *dual formulation* of the
Soft-SVM (also called C-SVM): $$\label{eq:10}
  \begin{split}
    \max_\alpha\quad &-\frac12\left\| {\sum\nolimits_i \alpha_iy^{(i)}x^{(i)}} \right\|^2 + \sum\nolimits_i \alpha_i\\
    \text{s.t.}\quad&\sum\nolimits_i \alpha_iy^{(i)}=0,\\
    &0 \le \alpha_i \le C,\ 1 \le i \le n.
  \end{split}$$ Comparing [\[eq:10\]](#eq:10){reference-type="eqref"
reference="eq:10"} with [\[eq:7\]](#eq:7){reference-type="eqref"
reference="eq:7"} we see the extra upper bound constraint
$\alpha_i \le C$. The complementary slackness conditions
for [\[eq:9\]](#eq:9){reference-type="eqref" reference="eq:9"} tell us
that $$\begin{aligned}
  \alpha_i = 0 & \implies\quad y^{(i)}(w^Tx^{(i)}+w_0) \ge 1\quad &(\text{correctly classified})\\
  \alpha_i = C & \implies\quad y^{(i)}(w^Tx^{(i)}+w_0) \le 1\quad &(\text{margin error})\\
  0 < \alpha_i < C & \implies y^{(i)}(w^Tx^{(i)}+w_0) = 1\quad &(\text{support vector}).
\end{aligned}$$

### The $\nu$-SVC$^\star$

In the C-SVM (either the primal or dual), even though we saw how the
parameter $C$ helps us trade off margin maximization with minimization
of training error, it is not easy to select this parameter, short of
cross-validation. There is an alternative formulation of the SVM that
makes the tradeoff easier to make. This is called the $\nu$-SVC (support
vector classifier), and is formulated as follows: $$\label{eq:11}
  \begin{split}
    \min_{w,\xi,w_0,\rho}\quad&\frac{1}{2}\left\| {w} \right\|^2 - \nu\rho + \frac{1}{n}\sum\nolimits_i \xi\\
    \text{s.t.}\quad& y^{(i)}(w^Tx^{(i)}+w_0) \ge \rho-\xi_i,\ \ 1 \le i \le n,\\
    &\xi_i \ge 0,\ 1\le i \le n,\ \ \rho \ge 0.
  \end{split}$$ In [\[eq:11\]](#eq:11){reference-type="eqref"
reference="eq:11"} we have two additional parameters: $\nu$, a
hyperparameter and $\rho$ a scaling of the margin that must be
optimized. What role does $\rho$ play? Consider a point for which
$\xi_i=0$, then its margin is $2\rho/\left\| {w} \right\|$. The
parameter $\nu$ upper bounds the margin error (that is, the fraction of
points that have $\xi_i > 0$). It can be shown that if $\rho > 0$, then
$\nu$ is an upper bound on the fraction of margin errors and a lower
bound on the fraction of support vectors. So this single parameter
allows a direct control on the two competing quantities in an SVM, and
given its interpretation, it may be easier to set.

We take a note that if $\nu$-SVC solution yields $\rho>0$, then it can
be shown that the usual C-SVM with $C=1/\rho$ leads to the same decision
function (hyperplane classifier).

## SVM as regularized risk minimization$\star$

We penalized the slacks linearly by adding $C\sum_i\xi_i$. One may ask
why not something else? Indeed, other choices have also been considered,
e.g., $C\sum_i\xi_i^p$ for $p\ge 1$. However, consider for the moment
what we may actually like to penalize, namely,
$$\|\xi\|_0 = \text{card}(\xi) = \#\text{nonzeros}(\xi),$$ which
corresponds to minimizing the number of training errors. Written in this
notation, we rewrite the SVM cost function
from [\[eq:7\]](#eq:7){reference-type="eqref" reference="eq:7"} as
$$\label{eq:12}
  \min_{w,w_0}\quad\frac{1}{2}\left\| {w} \right\|^2 + C\sum\nolimits_i \llbracket y^{(i)} \neq f(x^{(i)})\rrbracket.$$
The Iverson bracket above can be rewritten as the loss function
$$L(a,g)=L(y^{(i)},f(x^{(i)})) = L(y^{(i)}, \mathop{\mathrm{sgn}}(w^Tx^{(i)}+w_0)) =
  \llbracket y^{(i)}(w^Tx^{(i)}+w_0) \neq 1\rrbracket.$$ With this loss
function, the optimization
problem [\[eq:12\]](#eq:12){reference-type="eqref" reference="eq:12"} is
NP-Hard [@XYZ]. But by using the soft-margin SVM we make the problem
tractable. A brief observation shows that we can rewrite the soft-margin
SVM as a regularized loss (risk) minimization problem. Here's how.
Consider the constraint
$$y^{(i)}(w^Tx^{(i)}+w_0) \ge 1 - \xi_i,\quad\forall i,\ \xi_i \ge 0.$$
Since we are trying to make $\xi_i$ small (try to make the argument more
rigorous), we see that
$$\xi_i = \max(0,1-y^{(i)}(w^Tx^{(i)}+w_0)) = \max\ifmmode{\left\{ 0, 1 - y^{(i)}f(x^{(i)}) \right\}}\else\mbox{$\left\{ 0, 1 - y^{(i)}f(x^{(i)}) \right\}$}\fi.$$
Thus, we can write the C-SVM [\[eq:7\]](#eq:7){reference-type="eqref"
reference="eq:7"} equivalently in its *hinge-loss* formulation:
$$\label{eq:13}
  \min_{w,w_0}\quad\frac{1}{2}\left\| {w} \right\|^2 + C\sum\nolimits_i \max\ifmmode{\left\{ 0, 1 - y^{(i)}f(x^{(i)}) \right\}}\else\mbox{$\left\{ 0, 1 - y^{(i)}f(x^{(i)}) \right\}$}\fi.$$
The loss in [\[eq:13\]](#eq:13){reference-type="eqref"
reference="eq:13"} is called the *hinge loss* because of its shape (see
Fig. [7](#fig:loss){reference-type="ref" reference="fig:loss"}), and it
provide a convex upper bound to the nonconvex 0/1 loss used
in [\[eq:12\]](#eq:12){reference-type="eqref" reference="eq:12"}.

![Loss functions, denoted $\ell(z)=\max(0,1-z)$ for the hinge loss and
$\ell(z)=\log_2(1+e^{-z})$ for the logistic loss (for SVMs and LR
$z=y(w^Tx+w_0)$).](./svms7.png){#fig:loss width="0.4\\linewidth"}

Recall that logistic regression took on a form similar
to [\[eq:13\]](#eq:13){reference-type="eqref" reference="eq:13"},
namely,
$$\min_{w,w_0}\quad\frac{1}{2}\left\| {w} \right\|^2 + {\sf cross-entropy}(w,w_0),$$
where cross-entropy represents the loss function as in
[\[eq.crossent.1\]](#eq.crossent.1){reference-type="eqref"
reference="eq.crossent.1"}. In fact, more generally, $\ell_2$-norm
regularized risk minimization problems in ML are written as (we suppress
the bias term for simplicity):
$$\min_{w}\quad \frac{1}{2}\left\| {w} \right\|^2 + \frac{1}{n}\sum\nolimits_i L(y^{(i)}, f(x^{(i)})),$$
where $f(x^{(i)})$ is our "guess" at the true label. The separable (as a
sum over training data) structure of this regularized risk minimization
problem has made SGD like methods very popular for its solution.

## Additional remarks$\star$

We looked at the basic SVM formulation motivated by the geometric view
of seeking a large-margin classifier. Many more variations of the basic
theme exist, we have covered only a few key ideas. Some interesting
points worth knowing about are the following:

- Novelty detection via 1-class SVMs. This problem is actually an
    *unsupervised problem*, where we have a dataset drawn from some
    distribution $P$, and we wish to estimate a "simple" subset $S$ of
    the input space such that the probability that a test point from $P$
    lies outside the region $S$ equals some a priori specified value in
    $(0,1)$. For more details on this topic, the reader is referred to
    the paper "Support Vector Method for Novelty Detection" by Schölkopf
    et al. (NIPS 1999; vol 12).

- Running SGD on the hinge-loss formulation is a scalable method for
    dealing with large datasets. One popular way to solve this
    formulation is via an SGD variant called Pegasos (covered in
    Homework 2).

- There exist numerous strategies for obtaining $w_0$ (there is no
    unique bias, and various choices can be obtained from the KKT
    conditions). It is worth noting that in high-dimensional problems,
    the bias may not be that important.

- An idea key to the early success and adoption of SVMs was how
    researchers could incorporate prior (domain) knowledge into the
    formulation, as well as how they could encode invariances. These
    ideas proved crucial to making SVMs reliably outperform competition
    and become established as a fundamental classification method.

- We did not cover this point in class, but often one wishes to obtain
    probabilities from an SVM output. There have been several approaches
    in the literature for extracting probabilities, i.e., $P(Y|X)$ from
    the SVM parameters. One of the most popular choices is "Platt
    scaling" (see Wikipedia for a detailed description). There exist
    other more recent, theoretically better grounded alternatives, e.g.,
    "Support Vector Machines as Probabilistic Models" by Franc, Zien,
    Schölkopf (ICML 2011).

- Unsurprisingly, researchers have also considered endowing SVMs with
    a Bayesian viewpoint. There is reasonably substantive literature
    available on this topic, with very recent works (dating to 2014).

- SVMs have contributed extensively to the growth of interest in
    optimization algorithms with machine learning, and they have been
    solved using numerous techniques by now. Coordinate descent on the
    dual is a very popular strategy, and is used in the popular libary
    LIBSVM. For the primal, Pegasos is a popular approach too.

- Multi-class versions of the SVM are also possible, however we omit
    discussing these.

# Linear models with nonlinear features

We are now ready to discuss nonlinear classification. Indeed, the
ability to efficiently perform nonlinear classification was one of the
factors that contributed to the SVM's early success. From our
presentation above, it may not be apparent (to the uninitiated) how one
can perform efficient nonlinear classification using SVMs. This section
outlines the principal details of this idea. At a high level, it is the
flipped story of nearest neighbor: recall that the nearest neighbor
classifier is a nonlinear classifier with given features, whereas SVM
can perform nonlinear classification using linear classifier with
nonlinear features.

## Nonlinear feature maps

An idea that we have also encountered in previous lectures is that of
using nonlinear features. The idea here is to map the input data into a
(potentially) high dimensional space, and then run the usual linear
separators on the new representation. Which map one uses to transform
the input data nonlinearly depends on the domain, and is in general a
hard question.

![Effect of the nonlinear map $\phi: \mathbb{R}\to \mathbb{R}^2$ given
by $x \mapsto (x, |x|)$](./kernFig3.png){#fig:nonlin1
width=".7\\linewidth"}

![Effect of the nonlinear map $\phi: \mathbb{R}^2 \to \mathbb{R}^3$
given by $x \mapsto (x_1, x_2, x_1x_2)$](./kernFig4.png){#fig:nonlin2
width=".7\\linewidth"}

Figures [8](#fig:nonlin1){reference-type="ref" reference="fig:nonlin1"}
and [9](#fig:nonlin2){reference-type="ref" reference="fig:nonlin2"}
illustrate how patterns that are not linearly separable can become
linearly separable after creating nonlinear features. Notice how easy it
becomes to separate the XOR data
(Fig. [9](#fig:nonlin2){reference-type="ref" reference="fig:nonlin2"})
by embedding it into $\mathbb{R}^3$. In general, it is either
trial-and-error, or an art on how to create nonlinear maps that embed
data into (typically) high dimensional space where they become linearly
separable. You can imagine a variety of such nonlinear maps: the crux
being that we still continue to use linear separators, but linear in the
nonlinear features $\phi(x)$. In the next lecture, we will see a
powerful way to construct nonlinear features using neural networks.

It is important to reflect at this point on the computational and
storage ramifications of constructing the nonlinear maps. Typically, we
use $\phi: \mathbb{R}^d \to \mathbb{R}^m$, where $m \gg d$ (we can even
have $m=\infty$!). In this case, the cost of transforming input data
into vectors in $\mathbb{R}^m$ can be very high. Moreover, storing such
long vectors is also very expensive, as is the cost of any operation
(e.g., $w^T\phi(x)$) that we perform on such data. Remarkably, it turns
out that in many cases:

*The mapping $\phi(x)$ need not be explicitly performed!*

This remarkable idea brings us to the main topic of this section:
**kernel functions**.

## Kernel functions

The key assumption here is that we have access to a function that can
compute the inner product $\langle \phi(x^{(i)}), \phi(x^{(j)})\rangle$
without having to explicitly construct the "coordinates." While this
idea may seem mysterious at first sight, a few examples help illustrate
the point. **Example:** Consider the feature map
$\phi: \mathbb{R}^2 \to \mathbb{R}^3$ given by
$$\phi(x) = (x_1^2, \sqrt{2}x_1x_2, x_2^2).$$ If we compute the inner
product between two points $x, z \in \mathbb{R}^2$ by first forming
$\phi(x)$ and $\phi(z)$, we obtain two vectors of length 3 each. Thus,
computing $\langle \phi(x), \phi(z)\rangle$ will cost us $3$
multiplications and 2 additions. However, observe that $$\begin{split}
   \langle \phi(x), \phi(z)\rangle &= (x_1^2z_1^2+ 2x_1x_2z_1z_2+x_2^2z_2^2)\\
   &= (x_1z_1+x_2z_2)^2 = \langle x, z\rangle^2.
 \end{split}$$ Notice that to compute the inner product
$\langle \phi(x), \phi(z)\rangle$, we do *not need* to compute the
vectors $\phi(x)$ and $\phi(z)$. We simply compute $\langle x, z\rangle$
(at a cost of 2 multiplications and 1 addition) and then square it.
Thus, in this case we have $k(x,z)=\langle x, z\rangle^2$.

**Example:** Consider the kernel $k(x,z)=\langle x, z\rangle^m$ defined
on vectors $x, z \in \mathbb{R}^d$. To compute $k(x,z)$ we require only
$d-1$ multiplications to evaluate $\langle x, z\rangle$ and the cost to
compute its $m$-th power. However, if we were to instead first compute
the nonlinear map
$$\phi(x) = (x_{i_1}x_{i_2}\ldots x_{i_m})\quad 1\le i_1,i_2,\ldots,i_m \le d,$$
we would require much larger storage and much larger (how much?) amount
of computation to perform $\langle \phi(x), \phi(z)\rangle$ explicitly.

### Implications for SVM

To solve the dual of the SVM, we assume that we have access to a kernel
function $k$ that computes
$k(x^{(i)},x^{(j)})=\langle{\phi(x^{(i)})},\phi(x^{(j)})\rangle$.
Introducing the *kernel matrices* $K$ and $\hat{K}$ with entries given
by $$\label{eq:17}
 K_{ij} := k(x^{(i)},x^{(j)}),\quad 1\le i,j\le n,\qquad \hat{K}_{ij} = y^{(i)}K_{ij}y^{(j)},$$
respectively, and the all ones vector $\bm{1}_n\in \mathbb{R}^n$, the
dual of the SVM becomes $$\label{eq:14}
 \begin{split}
   \max_{\alpha\in \mathbb{R}^n}\quad &-\frac{1}{2}\alpha^T\hat{K}\alpha + \alpha^T\bm{1}_n\\
   &\alpha^Ty = 0,\quad \alpha \in [0,C].
 \end{split}$$ Observe that for
problem [\[eq:14\]](#eq:14){reference-type="eqref" reference="eq:14"} to
be convex, the matrix $\hat{K}$ must be positive (semi)definite. This
matrix will be positive (semi)definite if $K$ is positive
(semi)definite. And we will soon see that this positive definiteness
holds because of how we define a kernel function. Even though we have
"kernelized" the dual, what we are truly interested in is the
classifier, namely
$$w = \sum\nolimits_i \alpha_i y^{(i)}\phi(x^{(i)}).$$ Naively, this
classifier still requires high storage because it involves long vectors
$\phi(x^{(i)})$. However, since we are using a linear classifier in the
$\phi$ space, we can simply compute $$\begin{aligned}
 \mathop{\mathrm{sgn}}(w^T\phi(x)+w_0 ) &= \mathop{\mathrm{sgn}}\left(\sum\nolimits_i \alpha_i y^{(i)}\langle \phi(x^{(i)}), \phi(x)\rangle + w_0 \right)\\
 &=\mathop{\mathrm{sgn}}\bigl(\sum\nolimits_i \alpha_i y^{(i)}k(x^{(i)}, x) + w_0 \bigr),
\end{aligned}$$ which involves only computation of the kernel function.

## Kernel ridge regression$\star$

Consider the penalized empirical risk for ridge regression (RR) using
nonlinear features: $$\label{eq:15}
 R_n(w) := \frac{1}{n}\sum_{i=1}^n (y^{(i)}-\langle w, \phi(x^{(i)})\rangle)^2 + \frac\lambda2\left\| {w} \right\|^2.$$
We assume below that $\lambda > 0$. Our aim is to show how, like the
kernelized SVM from the previous section, we can
solve [\[eq:15\]](#eq:15){reference-type="eqref" reference="eq:15"}
without explicitly computing or storing $\phi(\cdot)$. Differentiating
$R_n(w)$ we must solve
$$\frac{\partial R_n(w)}{\partial w} = 0,\quad\implies \lambda w - \frac1n\sum_i \underbrace{(y^{(i)}-\langle w, \phi(x^{(i)})\rangle)}_{=: n\lambda \alpha_i}\phi(x^{(i)}) = 0,$$
which we can rewrite as
$$\lambda w = -\lambda \sum\nolimits_i \alpha_i \phi(x^{(i)}),\quad \implies w = \sum\nolimits_i \alpha_i \phi(x^{(i)}),$$
a form that holds provided for all $j$ we can find $\alpha_j$ such that
$$\label{eq:18}
 n\lambda \alpha_j = y^{(j)}-\langle w, \phi(x^{(j)})\rangle\qquad 1\le j \le n.$$
Note that now our predictions are
$$f(x) = \sum\nolimits_i \alpha_i \langle \phi(x^{(i)}), \phi(x)\rangle = \sum\nolimits_i \alpha_i k(x^{(i)},x),$$
which requires just calls to the kernel function. It remains to show
that we can find suitable $\alpha_i$ values. Plugging in the formula for
$w$ into [\[eq:18\]](#eq:18){reference-type="eqref" reference="eq:18"}
we obtain
$$n\lambda \alpha_j = y^{(j)} - \sum\nolimits_i \alpha_i \langle \phi(x^{(i)}), \phi(x^{(j)})\rangle = y^{(j)} -\sum\nolimits_i \alpha_i k(x^{(i)},x^{(j)}).$$
Writing the matrix $K=[k(x^{(i)},x^{(j)})]_{i,j=1}^n$ we obtain a large
($n\times n$) linear system
$$n\lambda \alpha = y - K\alpha\quad\implies\quad \alpha = (K+n\lambda I)^{-1}y.$$
Thus, we have managed to kernelize ridge regression. Importantly, note
the similarities between the kernel SVM predictor and the Kernel RR
predictor $$\begin{aligned}
{2}
 \text{SVM}:\quad & w = \sum\nolimits_i \alpha_iy^{(i)}\phi(x^{(i)}),\quad&\alpha_i\ \text{dual variables}\\
 \text{KRR}:\quad & w = \sum\nolimits_i \alpha_i\phi(x^{(i)}),\quad&\alpha=(K+n\lambda I)^{-1}y.
\end{aligned}$$ The most useful fact about representing $w$ as a linear
combination of the $\phi(x^{(i)})$ (for SVM, KRR, or any other model
where we have such a representation) is that the prediction also reduces
to being able to evaluate the kernel function since
$w^T\phi(x)=\sum_i\alpha_ik(x^{(i)}, x)$.

This similarity between the SVM and KRR predictors is not a coincidence,
but part of a much more general result about regularized problems
involving kernels, which roughly says that even though the feature
vectors $\phi(x^{(i)})$ are potentially infinite dimensional (and
therefore, as is our predictor $w$), the intrinsic dimension of our
predictor is at most $n$ as we can represent it as a linear combination
of the $n$ training data points. This claim is made more precise in the
section below.

## The representer theorem$\star$

We state here an important theorem that presents a general setting under
which we can expect the predictor to be a linear combination of the
input training data. Initial aspects of this theorem (for the squared
loss) were presented in a seminal work of Wahba et al. in the 1970s;
much later, Schölkopf, Herbrich, and Smola presented sufficient
conditions in 2001. However, only recently in 2013 was the version noted
below (which contains both necessary and sufficient conditions) was
discovered by Yu et al (ICML 2013).

::: {#thm:rep .theorem}
*Let $x^{(1)},\ldots,x^{(n)}$ be training data drawn from
a set $\mathcal{X}$. Let $R_n$ (condition on $R_n$?) be some loss
function from $\mathbb{R}^n \to \mathbb{R}$, and $\mathcal{H}$ the RKHS
induced by the kernel $k : \mathcal{X}\times \mathcal{X}\to \mathbb{R}$,
with induced norm $\|f\| := \sqrt{\langle f, f\rangle}$. Let
$\Omega: \mathcal{H}\to \mathbb{R}\cup\ifmmode{\left\{ \infty \right\}}\else\mbox{$\left\{ \infty \right\}$}\fi$.
Consider the penalized empirical risk minimization problem
$$\label{eq:16}
   \min_{f \in \mathcal{H}}\quad R_n(f(x^{(1)}),\ldots, f(x^{(n)})) + \lambda\Omega(f).$$
Then, at least one of the minimizers
of [\[eq:16\]](#eq:16){reference-type="eqref" reference="eq:16"} has the
representation $w = f(\cdot) = \sum_i \alpha_i k(\cdot, x^{(i)})$ for
some $\alpha \in \mathbb{R}^n$ if and only if
$$\forall f, g \in \mathcal{H},\quad \left\| {f} \right\| > \left\| {g} \right\| \implies \Omega(f) \ge \Omega(g).$$
Moreover, if $\Omega$ is strictly increasing then all the minimizers
have this representation.*
:::

In other words, if we wish our predictor to enjoy a linear
representation in terms of training data, we are forced to use a
regularizer that depends monotonically on the Hilbert space norm of the
predictor. This result immediately implies the representations we saw
for Kernel-SVM and Kernel RR; using it we can obtain similar conclusions
with all kinds of other loss functions. It just relies on the kernel $k$
generating an RKHS.

# Basic theory of kernels

The alert reader may have noticed that when writing out the representer
theorem, as well as when using kernels we did not require the kernel
function to be defined over vectors. We permitted any set $\mathcal{X}$.
Indeed, this ability to seamlessly work with non-vector valued data, as
long as we have a "similarity function" (that is, a kernel) over pairs
available, we can perform the usual classification and regression over
such data. This powerful ability of kernel methods was also an early
contributor to their widespread adoption, e.g., in bioinformatics where
researchers were working with genetic data encoded as strings, or
phylogenetic trees; in computational chemistry, where the data were
small graphs / networks of molecules, etc. Let us formalize the notion
of a kernel function below. Let $\mathcal{X}$ be any nonempty set, and
let $x^{(1)},\ldots,x^{(n)}$ be any set of $n$ elements drawn from
$\mathcal{X}$. A kernel function is a map
$$k: \mathcal{X}\times \mathcal{X}\to \mathbb{R},$$ that satisfies a few
important properties noted below. In particular, we will reserve the
term "kernel function" for a *positive definite (PD) kernel*, also known
as *Mercer kernel*, or henceforth, simply a "kernel." **Kernel
function:** Specifically, a map
$k: \mathcal{X}\times \mathcal{X}\to \mathbb{R}$ is called a kernel
function if there exists a Hilbert space $\mathcal{H}$ and a feature map
$\phi: \mathcal{X}\to \mathcal{H}$ such that

1. The map $k$ is symmetric, i.e., $k(x,x')=k(x',x)$ for all
    $x,x'\in\mathcal{X}$

2. For all $x, x' \in \mathcal{X}$, we have
    $k(x,x')=\langle \phi(x), \phi(x')\rangle$,

3. For all $n \in \mathbb{N}$, for arbitrary
    $x^{(1)},\ldots,x^{(n)}\in \mathcal{X}$, the $n\times n$ matrix
    $$K = [k(x^{(i)},x^{(j)})]_{i,j=1}^n,$$ is positive semidefinite.

Please note that the map $\phi$ *need not* be unique---corresponding to
a given kernel function, there can be an infinite number of different
feature maps $\phi$ such that
$k(x,x')=\langle \phi(x), \phi(x')\rangle$. Given the above definition
of kernel functions, by using properties of positive definite matrices,
we can conclude some important algebraic properties of kernel functions
that allow us to construct new kernel functions by combining old ones:

- **(CC).** If $k_1$ and $k_2$ are kernels, then so is $ak_1+bk_2$ for
    scalars $a,b \ge 0$

- **(PP).** If $k_1$ and $k_2$ are kernels, then so is $k_1k_2$.

Of these, property (CC) is immediate because the nonnegative sum of two
positive definite matrices is again positive definite. Property (PP) is
a consequence of the famous "Schur product" theorem:

{: .theorem }
*Let $A, B$ be two symmetric positive definite matrices.
Then, the matrix $C$ whose entries are defined by $c_{ij}:=a_{ij}b_{ij}$
(denoted as $C=A\circ B$) is also symmetric positive definite.*

Using these two properties we can easily obtain new kernels. The
following examples are immediate:

1. $k(x,x')=\langle x, x'\rangle$ (Linear kernel); simply from
    definition

2. $\langle x, x'\rangle^m$ (polynomial kernel); inductively use the
    product property PP.

3. $e^{\langle x, x'\rangle} = 1 + \langle x, x'\rangle + \frac{\langle x, x'\rangle^2}{2!}+\cdots$
    (exp kernel); use CC and PP

4. $e^{-\gamma\|x-x'\|^2}$ (Gaussian RBF kernel; $\gamma > 0$) use PP
    and (iii).

5. Challenge: Is $\frac{1}{1+\left\| {x-x'} \right\|}$ a kernel? What
    about $e^{-\gamma\left\| {x-x'} \right\|}$?

## Additional examples$^\star$

We include below some interesting examples of kernels. We do not include
proofs of these facts. The interested reader is encouraged to attempt
proving them on their own.

- **Probability kernel**: Let $(\Omega, P)$ be a probability space
    with measure $P$. Then for $A, B \in \Omega$ the function
    $$k(A,B) := P(A \cap B) - P(A)P(B),$$ is a kernel.

- **Jaccard kernel**: This kernel is very useful for computing a
    similarity between arbitrary sets. For arbitrary nonempty sets
    $A, B$ it is defined by $$k(A,B) = \frac{|A\cap B|}{|A\cup B|}.$$

- **String kernel**: Let $x \in \mathcal{A}^d$ and
    $x' \in \mathcal{A}^{d'}$ be two strings on the alphabet
    $\mathcal{A}$, of length $d$ and $d'$, respectively. Then,
    $$k(x,x') := \sum_{S\in \mathcal{A}^*} w_S\phi_S(x)\phi_S(x'),$$ is
    a kernel, where $\mathcal{A}^*$ is the set of all finite strings
    over the alphabet, $w_S$ is a nonnegative weight function of the
    string $S$, and $\phi_S(x)$ is a feature map whose coordinate
    corresponding to $S$ is $1$ if the substring $S$ occurs in $x$ (this
    feature map can be viewed as a fancy way of doing 1-hot encoding).

- **Graph kernels:** This is a topic of even research interest. Some
    kernels in this domain are *graphlet kernels* (obtained by
    enumerating subgraphs of up to a certain size and using feature maps
    akin to that in the string kernel), the *Weisfeiler-Lehman* kernel,
    amongst many others. The ideal graph kernel would help characterize
    graph isomorphism or subgraph isomorphism, but clearly, that would
    make the kernel rather impractical (subgraph isomorphism is
    NP-Hard).

- **Translation invariant kernels:** These are kernel functions of the
    form $k(x,x')=h(x-x')$ for a suitable function. It is known
    (Bochner's theorem) that a function $h(t)$ generates a translation
    invariant kernel if and only if its Fourier transform is
    nonnegative.

- **Tensor product kernel:** Let $k_1$ and $k_2$ be kernels overs
    $\mathcal{X}_1\times \mathcal{X}_1$ and
    $\mathcal{X}_2\times \mathcal{X}_2$, respectively. Then, their
    tensor product
    $k_1\otimes k_2(x_1,x_2,x_1',x_2')=k_1(x_1,x_1')k_2(x_2,x_2')$ is a
    kernel on the product space
    $(\mathcal{X}_1\times \mathcal{X}_2) \times (\mathcal{X}_1\times \mathcal{X}_2)$.

- **Kernels over kernel matrices:** These kernels have been used in
    computer vision and other areas where the input data is not
    vectorial, but covariance matrices (e.g., graph Laplacians, kernel
    matrices, etc.). For example, if $A, B$ are two positive definite
    matrices, then
    $$k(A,B) := \frac{1}{\det(A+B)^{\alpha}},\quad \alpha \in \ifmmode{\left\{ \tfrac{j}{2} \cup s \mid 1 \le j \le m-1, s \ge \tfrac{m-1}{2} \right\}}\else\mbox{$\left\{ \tfrac{j}{2} \cup s \mid 1 \le j \le m-1, s \ge \tfrac{m-1}{2} \right\}$}\fi,$$
    is a kernel function (proving this is not easy).

- **Monotonic kernels:** Let $k(x,x')=h(x+y)$, where $h$ is given by
    the Laplace transform of a nonnegative measure. Then $k$ is a
    kernel. For example, $k(x,x') = 1/(x+x')^\alpha$ is a kernel for
    positive numbers $x,x'$ and $\alpha \ge 0$.

## Concluding remarks$\star$

An important question that you may wonder about is: *how should we
choose a kernel?* There is no clear answer to this question, as it
depends on the application. If we have non vector valued data, then
either we could first encode it using vectors (e.g., 1-hot encoding) and
then use a kernel over vectors, or we could directly define a kernel
over non-vectorial data (easier to interpret and use the results). Even
if our data are vectors, out of the vast choice of kernels which ones
should we use, and with what parameters?

One option is to use a *universal kernel*, like the Gaussian RBF (see
e.g., the paper "Universal kernels" by Michelli, Xu, and Zhang (JMLR
2006)). There are also arguments based on generalization properties of
different kernels. However, typically, the choice is application
dependent, and the user should tune the choice of the kernel to the
application. The choice of kernel reflects our "bias" or *prior*
knowledge about the application.

A final remark about kernels that we make in passing is that storing a
large $n\times n$ kernel matrix is not feasible for big-data problems.
Although we did not cover the topics in the lectures, on how to scale
kernel methods to problems where $n$ is large, we mention here the
Nyström approximation as a popular way to scale up kernel methods. Many
other methods exist that are based on approximating the kernel matrix,
or the feature map. The reader is encouraged to consult the wider
literature on these topics if they are interested in scaling up kernel
methods (which remains a topic of current research interest).

# Reproducing Kernel Hilbert Space$^\star$

The material in this section is optional, but we encourage you to read
through it. It provides a superficial sketch of how we build the
Reproducing Kernel Hilbert Space, better known as RKHS, corresponding to
a kernel function. Think of $\phi$ as a map that takes $x$ to the space
of real valued functions on $x$. Thus,
$$\phi \equiv x \mapsto \mathbb{R}^{\mathcal{X}} = \ifmmode{\left\{ f: \mathcal{X}\to \mathbb{R} \right\}}\else\mbox{$\left\{ f: \mathcal{X}\to \mathbb{R} \right\}$}\fi.$$
Thus, $x$ is mapped to a function parameterized by $x$, we write this as
$x \mapsto k(\cdot,x)$. In other words, $\phi(x)$ denotes the function
that assigns the value $k(x',x)$ to a point $x' \in \mathcal{X}$. Thus,
$$(\cdot) = k(\cdot, x),$$ where the ("long vector") $\phi(x)$ is
actually a function. Thus, each pattern in the input data gets mapped
into a function, and we may consider that each pattern is now
represented by its similarity to *all other points in the domain!* This
representation seems quite rich. Now, consider the set of functions
obtained from training data:
$\ifmmode{\left\{ \phi(x^{(1)}),\ldots \phi(x^{(n)}) \right\}}\else\mbox{$\left\{ \phi(x^{(1)}),\ldots \phi(x^{(n)}) \right\}$}\fi$
for $n \in \mathbb{N}$. We turn this set into a vector space and endow
this space with a dot product. Then, we show that this dot product
actually satisfies $k(x,x')=\langle \phi(x), \phi(x')\rangle$.

Once we have performed the above construction, then we see that for all
functions $\psi$ in the span of
$\ifmmode{\left\{ \phi(x^{(1)}),\ldots \phi(x^{(n)}) \right\}}\else\mbox{$\left\{ \phi(x^{(1)}),\ldots \phi(x^{(n)}) \right\}$}\fi$
(i.e., in the vector space we constructed), we have the *reproducing
property*: $$\langle k(\cdot,x), \psi\rangle=\psi(x).$$ This leads us to
formally define an RKHS as follows.\
**RKHS.** Let $\mathcal{H}$ be a Hilbert space of functions
$f:\mathcal{X}\to \mathbb{R}$. Then, $\mathcal{H}$ is called an RKHS if
there exists a function $k: \mathcal{X}\times \mathcal{X}\to \mathbb{R}$
with the following properties:

1. $k$ has the reproducing property:
    $\langle \psi, k(\cdot,x)\rangle=\psi(x)$ for all
    $\psi\in \mathcal{H}$

2. In particular, $\langle k(\cdot,x), k(\cdot,x')\rangle=k(x,x')$,

3. $k$ spans $\mathcal{H}$, i.e.,
    $\mathcal{H}= \overline{\text{span}\ifmmode{\left\{ k(\cdot,x) \mid x \in \mathcal{X} \right\}}\else\mbox{$\left\{ k(\cdot,x) \mid x \in \mathcal{X} \right\}$}\fi}$

A key aspect of RKHS's is that it uniquely determines the kernel $k$.
That is, for each kernel there is a unique RKHS where the kernel has the
reproducing property. Note that this just means that $\mathcal{H}$ is
unique; the map $\phi$ need not be unique.

[^1]: Half-spaces, hyperplanes, logistic regression classifier, etc.,
    all are instances of a linear hypothesis class composed with a
    suitable scalar function to extract the decision.

[^2]: A good reference to familiarize with KKT conditions, Lagrangians,
    and duality is Chapter 5 of "Convex Optimization" by Boyd and
    Vandenberghe. Another accessible source is "Lagrange multipliers
    without permanent scarring" by Dan Klein.
