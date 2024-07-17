---
title: Policy Gradient
nav_order: 5
---


# Policy Gradient Overview 
In Q learning, the Q function requires a state-action pair as its input. But
policy itself is only a function of state, and hence may be simpler. 

So we may model the policy itself directly using a parametric approach. Typically the policy is modeled as a distribution (in the exponential family); this modelling choice has two benefits:
- elegant differentiability
- stochastic policy promotes exploration 

<figure id="fig:policygrad">
<img src="../figs/policygrad.png" />
<figcaption>A parametric policy function.</figcaption>
</figure>

More specifically, our goal is to maximize the expected cumulative reward

$$\theta^*=\mathop{\arg\max}_\theta V(\theta)=\mathop{\arg\max}_\theta\mathbb{E}\big[\sum_{t=1}^\infty\gamma^tR(s_t,a_t)\big],$$


where we assume

- $s_0$ is from a distribution $\rho(\cdot)$;

- $a_t$ is from a distribution (policy_ $\pi_\theta(\cdot \|s_t;\theta)$;    
- $s_{t+1}$ is from a distribution given by the environment
  $p(\cdot|s_t,a_t)$.

A natural way to solve this maximization is to do gradient ascent
$\theta\leftarrow\theta+\alpha\nabla_\theta V(\theta).$ But how can we
compute the gradient $\nabla_\theta V(\theta)$? 

## the log trick
The way is to re-write $V(\theta)$ as an expectation with respect to trajectories. We define $\tau$ as a random variable of all states and actions: $$\tau=(s_0,_0;s_1,a_1;\dots).$$

Then the probability distribution of $\tau$ is given by

$$p(\tau;\theta)=\rho(s_0)\prod_{t\geq 0}p(s_{t+1}|s_t,a_t)\pi(s_t,a_t;\theta),$$

and the cumulative reward we obtain from this trajectory is

$$r(\tau)=\sum_{t\geq 0}\gamma^tR(s_t,a_t).$$

So we have $V(\theta)=\sum_\tau r(\tau)p(\tau;\theta).$ Then the gradient can be written as

$$\nabla_\theta V(\theta)=\nabla_\theta\sum_\tau r(\tau)p(\tau;\theta)=\sum_\tau r(\tau)\nabla_\theta p(\tau;\theta).$$

Note that we have

$$\nabla_\theta p(\tau;\theta)=p(\tau;\theta)\frac{\nabla_\theta p(\tau;\theta)}{p(\tau;\theta)}=p(\tau;\theta)\nabla_\theta\log p(\tau;\theta).$$

We also know that

$$\log p(\tau;\theta)=\log\rho(s_0)+\sum_{t\geq 0}\log p(s_{t+1}|s_t,a_t)+\sum_{t\geq 0}\log\pi(s_t,a_t;\theta).$$

The first two terms on the RHS do not depend on $\theta$, so they have
zero gradient, which means

$$\nabla_\theta\log p(\tau;\theta)=\sum_{t\geq 0}\nabla_\theta\log\pi(s_t,a_t;\theta).$$

So we have

$$\nabla_\theta V(\theta)=\sum_\tau r(\tau)p(\tau;\theta)\sum_{t\geq 0}\nabla_\theta\log\pi(s_t,a_t;\theta)=\mathbb{E}_{\tau\sim p(\tau;\theta)}\big[r(\tau)\sum_{t\geq 0}\nabla_\theta\log\pi(s_t,a_t;\theta)\big],$$

where the expectation can be estimated by sampling. 

## REINFORCE
So the following REINFORCE procedure is used to update policy using policy gradient:

- Sample trajectories $\tau_1,\tau_2, \dots, \tau_N$ as per $p(\tau;\theta)$.

- Estimate gradient as $\nabla_\theta V(\theta)\approx\frac{1}{N}\sum_{i=1}^N r(\tau_i)\big[\sum_{(s_t,a_t)\in\tau_i}\nabla_\theta\log\pi(s_t,a_t;\theta)\big].$

- Gradient ascent $\theta\leftarrow\theta+\alpha\nabla_\theta V(\theta).$

An variation of REINFORCE is to use

$$\nabla_\theta V(\theta)\approx\sum_{t\geq 0} \big(\sum_{t'\geq t}\gamma^{t'-t}R(s_{t'},a_{t'})\big)\nabla_\theta\log\pi(s_t,a_t;\theta)$$

as an approximation of

$$\nabla_\theta V(\theta)\approx\frac{1}{N}\sum_{i=1}^N r(\tau_i)\big[\sum_{(s_t,a_t)\in\tau_i}\nabla_\theta\log\pi(s_t,a_t;\theta)\big].$$

Intuitively, $\nabla_\theta V(\theta)$ changes the parameter $\theta$ so that if actions taken from a policy bring more reward, then increase its probability and decrease it otherwise.

We can also reduce the variance of REINFORCE by replacing the $\big(\sum_{t'\geq t}\gamma^{t'-t}R(s_{t'},a_{t'})\big)$ term by the advantage function 

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).$$ 

So we have 

$$\nabla_\theta V(\theta)\approx\sum_{t\geq 0} \big(Q^\pi(s,a)-V^\pi(s)\big)\nabla_\theta\log\pi(s_t,a_t;\theta).$$

## Actor-critic

To estimate the advantage $A^\pi(s,a)$, we introduce another critic model (parameterized as $\phi$) and use the following actor-critic algorithm.

<figure id="fig:actor_critic">
<img src="../figs/actor_critic.png" />
<figcaption>Actor-Critic Algorithm</figcaption>
</figure>