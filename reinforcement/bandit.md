---
title: Bandits
nav_order: 2
---

# Interpretations and Connections
## Single-state MDP

Recall the four types of decision making problems, described [here](/reinforcement/#sub-categories-of-decision-making).

Let's take a look at this scenario, where the model is unknown but the state is still fixed.

A simple example of such a setting is that we are managing an e-commerce web-portal. We want our customers to like the contents of the website so we need to decide which of the $N$ landing pages to show to a new customer. Different from the previous scenario, this time we have no idea **a priori** about how customers would react to each of the $N$ options and we do not know anything about any incoming customer.

Of course we need to keep exploring different options and build knowledge about our customer, but in the same time, we want to collect reward if we have a good strategy. So the problem is how can we balance between a good reward and at the same time, explore well.

Formally, we have

- Actions $A$ ($N$ landing pages), also known as Arms

- States $S$, which is a single state (null) and therefore the
  transition kernel is trivial

- Reward $R$, which is whether customers like or dislike the pages shown
  to them, noted as

$$\mathbb{E}[R(\text{null},a)] = \mu_a, \forall a \in A$$

Initially we have no idea any of the $\mu_a$ and by exploring the
actions, the reward of each of them will not be changed. Suppose we have
only finite chances to pull the arms, and our goal is to maximize the
cumulative reward over the time:

$$\text{maximize } \mathbb{E}[\sum_{t=1}^T R(a_t, \text{null})] = \sum_{t=1}^T \mu_{a_t}$$

Ideally, the optimal policy will be to always choose the best action
$a^*$ such that 

$$a^* \in \arg \max_{a\in A} \mu_a$$ 

However, we do not know any of the $\mu_a$ and therefore we need to discover it and the question is that how can we explore it without loosing too much time.

## Exploration vs exploitation

From another perspective, there’s a dilemma between to explore and to exploit. By exploring, we can find the expected reward of each action (arm), but this required enough samples for each of the actions. On the other hand, if an arm of a low reward is pulled too often, we are losing rewards from some other better options. So the challenge is how to achieve balance between explore and exploit so that we can pull sub-optimal arm as little as possible.

## Upper confidence bound (UCB)

To keep things simple, lets assume reward of each arm is between $[0,1]$, if an arm $i$ is pulled $k$ times, the empirical mean reward will be 

$$\hat{\mu}_i(k)=\frac{1}{k}\sum_{j=1}^k R_i(j).$$ 

By Hoeffding’s inequality, we know this estimation cannot be too far from the true value:

$$\mathbb{P}(|\hat{\mu}_i(k) - \mu_i| > \gamma) \leq 2 \exp(-\frac{k \gamma^2}{2})$$

In other words, with probability at least $1-\delta$,

$$\mu_i < \hat{\mu}_i(k) + \sqrt{\frac{2\log (\frac{2}{\delta})}{k}}$$

This means, with high probability ($1-\delta$), if we test arm $i$ for $k$ times, then the true expected reward of this arm is upper bounded by the empirical reward plus some controlling term involving $\delta$ and $k$. The problem now is how to choose $\delta$ and $k$.

The upper confidence bound algorithm gives a strategy on how to choose
arms:

- Initially, pick each arm once

- After that, for each time $t$, choose arm (ties broking arbitrarily)
  
$$I_t \in \arg \max \hat{\mu}_i(T_i(t-1)) + \sqrt{\frac{4 \log t}{T_i(t-1)}}$$

where $T_i(s)$ is the number of times arm $i$ is pulled till time $s$, so that $\hat{\mu}_i(T_i(t-1))$ is the empirical estimator of the reward for arm $i$ at time $t-1$. The second term gives us the bound of the difference between empirical and true expected reward with $\delta = \frac{2}{t^2}$

Intuitively, at each time, we are choosing the arm with the highest upper bound of their expected reward, where the upper bounds for every arm is computed such that it would be broken with very small probability. If the empirical reward for a certain arm is high, then it will be selected more to maximize the overall reward, but as it is chosen more, the second term, which is the believed difference between empirical and true expected reward for other arms, will be relatively larger and thus we need to explore other arms as well.


By doing so, we have a theorem saying that for any time $T$, the average number of times that any non-optimal arm is chosen is upper bounded by $\frac{16\|A\|\log T}{\Delta^2} + O(1)$ where $\Delta$ is the gap between average reward of the best and the second best arm, and $\|A\|$ is the number of arms we have.

To explain why this theorem holds, let’s keep the Hoeffding bounds in mind. Hoeffding says, with high probability,

$$\hat{\mu_i}(T_i(t-1)) < \mu_i + \sqrt{\frac{4\log t}{T_i(t-1)}} \quad \text{ and } \quad \hat{\mu_i}(T_i(t-1)) > \mu_i - \sqrt{\frac{4\log t}{T_i(t-1)}}$$


Now, let’s prove the theorem with contradiction. Suppose we did pull an non-optimal arm $i$ too often, which is

$$T_i(t-1) \geq \frac{16 \log t}{\Delta^2}$$ 

then,

$$\Delta \geq 2\sqrt{\frac{4\log t }{T_i(t-1)}}$$ 

Let’s suppose $i^*$ is the optimal arm, then with the definition of $\Delta$, we have 

$$\mu_{i^*} \geq \mu_i +\Delta$$ 

We know by the Hoeffding’s Inequality,

$$\hat{\mu}_{i^*}(T_{i^*}(t-1))  + \sqrt{\frac{4\log t}{T_{i^*}(t-1)}} > \mu_{i^*}$$

On the other hand, 

$$\begin{aligned}
\mu_i+ \Delta & \geq \mu_i + 2\sqrt{\frac{4\log t }{T_i(t-1)}} \\
& > \hat{\mu_i}(T_i(t-1)) - \sqrt{\frac{4\log t }{T_i(t-1)}} + 2\sqrt{\frac{4\log t }{T_i(t-1)}} \\
& =  \hat{\mu_i}(T_i(t-1)) + \sqrt{\frac{4\log t }{T_i(t-1)}} 
\end{aligned}$$ 

and thus we have 

$$\hat{\mu}_{i^*}(T_{i^*}(t-1))  + \sqrt{\frac{4\log t}{T_{i^*}(t-1)}} > \hat{\mu_i}(T_i(t-1)) + \sqrt{\frac{4\log t }{T_i(t-1)}}$$

which means the upper confidence bound algorithm would not pick arm $i$ and would pick arm $i^*$ instead.

Therefore, for any arm $i$ which is not optimal, till any given time $t$, we would not pick it more than $\frac{16\log t}{\Delta^2}$ times and the total number we pick non-optimal arms would be bounded by $\frac{16\|A\| \log t}{\Delta^2},$ where $\|A\|$ is again the total number of arms.
