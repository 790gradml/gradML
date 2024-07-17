---
title: Value functions and Bellman
nav_order: 4
---

## Value Functions

### State Value Functions (V functions)

The value function for a given policy $\pi$ is $V^{\pi} : S \to \mathbb{R}$, defined as:

$$\begin{aligned}
V^{\pi}(s) = \mathbb{E}\left[ \sum_{t=0}^T \gamma^t R(s_t, \pi(s_t)) \Big| s_0 = s \right]
\end{aligned}$$

In words, $V^{\pi}(s)$ is the expected sum of discounted rewards obtained by starting in
state $s$ and following policy $\pi$.

{: .exercise}
> Think about what is this expectation taken with respect to?

### Bellman Equation
By expanding the terms and rearranging, we see that:

$$\begin{aligned}
V^{\pi}(s) &= \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, \pi(s_t)) \Big| s_0 = s \right]\\
&= R(s, \pi(s)) + \gamma \sum_{s' \in S} p(s'|s,\pi(s)) \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, \pi(s_t)) \Big| s_0 = s' \right]\\
&= R(s, \pi(s)) + \gamma \sum_{s' \in S} p(s'|s,\pi(s)) V^{\pi}(s')
\end{aligned}$$

The term $R(s, \pi(s))$ is the *immediate reward* (i.e. the expected reward, obtained at the first action). Whereas the term $\gamma \sum_{s' \in S} p(s'\|s,\pi(s)) V^{\pi}(s')$ is the *sum of discounted future reward* (i.e. the total reward obtained for all actions after the first action).


### Bellman Optimality Equation

We say that a policy $\pi^*$ is optimal if its value function is larger
at every state than the value function of any other policy:

$$V^{\pi^*}(s) \geq V^{\pi}(s), \quad \forall s \in S, \pi: S \to A$$

The immediate question to ask is: does an optimal policy exist? Theorem <a href="#thm:bellman" data-reference-type="ref" data-reference="thm:bellman">1</a> answers the question in the affirmative, and gives the *Bellman equation* which characterizes the value function of the optimal policy.

{: .theorem}
> **Theorem 1** (Bellman Equation). 
> Given a markov decision process $(S, A, p, R, \gamma)$, there exists an optimal policy $\pi^*$, and its value function satisfies:
> $$V^{\pi^*}(s) = \max_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi^*}(s') \right]$$

{: .proof}
>*Proof sketch..* Consider any policy $\pi: S \to A$. By definition, it satisfies:
>
>$$V^{\pi}(s) = R(s, \pi(s)) + \gamma \sum_{s' \in S} p(s'|s,\pi(s)) V^{\pi}(s')$$
>
>Consider an improved value function:
>
>$$V^{\pi'}(s) = \max_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi}(s') \right]$$
>Note that $V^{\pi'}$ may not correspond to an actual policy. 
>
>In any case, it is clear that:
$$\begin{aligned}
V^{\pi'}(s) \geq V^{\pi}(s), &\quad \forall s \in S\\
V^{\pi'}(s) > V^{\pi}(s), &\quad \forall s \in S \text{ such that } V^{\pi}(s) \text{ does not satisfy the Bellman equation}
\end{aligned}$$ 
>
>Therefore, as long as the Bellman equation is not satisfied for all states, it is possible to iteratively improve the value function. The maximum value that such a value function can attain is bounded (by $R_{max} \frac{1}{1-\gamma}$, where $R_{max} = \max_{s \in S, a \in A} R(s, a)$), so the sequence of value functions converges to a value function that satisfies the Bellman equation for each state: 
>
>$$V^{\pi^*}(s) = \max_{a \in A} \left[ R(s, \pi(s)) + \gamma \sum_{s' \in S} p(s'|s,\pi(s)) V^{\pi^*}(s') \right]$$ 
>
>Therefore, a value function that satisfies the Bellman equation exists, and it is easy to see that it corresponds to an actual policy $\pi^*$: 
>
>$$\pi^*(s) = \mathop{\mathrm{argmax}}_{a\in A} \left[ R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi^*}(s') \right]$$ 

So far, we showed that a fixed point exists to the Bellman equation, and that it corresponds to an actual policy $\pi^*$. Is this policy optimal? It is clear that an optimal policy needs to satisfy the Bellman equation; otherwise it would be possible to improve it by choosing the maximizing action at each state. Then, it is enough to show that the fixed point to the Bellman equation is unique. This will be shown in the discussion of value iteration. ◻

Note that, even if the optimal value function is unique, the optimal policy is not necessarily unique; for example, there may be multiple maximizing actions $a \in A$ for each state. 

### Value Iteration

Consider iteratively improving the value function given the improvement operation discussed in the proof to Theorem <a href="#thm:bellman" data-reference-type="ref" data-reference="thm:bellman">1</a>. 

That is, at iteration $k \geq 1$, let:

$$V^{\pi^{k+1}}(s) = \max_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi^{k}}(s') \right]$$

We analyze the evolution of 

$$\|\|V^{\pi^k}-V^{\pi^*}\|\|_\infty = \max_{s \in S} \|V^{\pi^k}(s)-V^{\pi^*}(s)\|$$.

We have: 

$$\begin{aligned}
& V^{\pi^{k+1}}(s) - V^{\pi^{*}}(s)\\
&= \left[ \max_{a\in A} R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi^k}(s') \right] - \left[ \max_{a'\in A} R(s,a') + \gamma \sum_{s' \in S} p(s'|s,a') V^{\pi^*}(s') \right]\\
&= \max_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi^k}(s') - \max_{a'\in A} \left( R(s,a') + \gamma \sum_{s' \in S} p(s'|s,a') V^{\pi^*}(s')\right) \right]\\
&\leq \max_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi^k}(s') - R(s,a) - \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi^*}(s')\right]\\
&\leq \gamma \max_{a \in A}  \sum_{s' \in S} p(s'|s,a) |V^{\pi^k}(s') - V^{\pi^*}(s')|\\
&\leq \gamma \max_{a \in A}  \sum_{s' \in S} p(s'|s,a) ||V^{\pi^k} - V^{\pi^*}||_\infty\\
&= \gamma ||V^{\pi^k} - V^{\pi^*}||_\infty
\end{aligned}$$

Similarly:

$$V^{\pi^{*}}(s) - V^{\pi^{k+1}}(s) \leq \gamma ||V^{\pi^k} - V^{\pi^*}||_\infty$$

Therefore:

$$||V^{\pi^{*}} - V^{\pi^{k+1}}||_\infty \leq \gamma ||V^{\pi^k} - V^{\pi^*}||_\infty$$

Since $0 \leq \gamma < 1$, we have therefore that $\|\|V^{\pi^k} - V^{\pi^*}\|\|_\infty \to 0$ as $k \to \infty$, starting from any initial condition.

Because this holds for any $$V^{\pi^*}$$ that satisfies the Bellman equation, an implication is that the value function $V^{\pi^*}$ that satisfies the Bellman equation is unique.

### Policy Iteration

Value iteration attempts to find the fixed point of the Bellman equation directly, by iterating over value functions. Alternatively, we could iterate over policies.

Start with any policy $\pi^0$. Then, iteratively evaluate the function $V^{\pi^k}$ for policy $\pi^k$, and select the updated policy $\pi^{k+1}$ as:

$$\pi^{k+1}(s) = \mathop{\mathrm{argmax}}_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi^k}(s') \right]$$

To evaluate the function $V^{\pi^k}$, one needs to solve:

$$V^{\pi}(s) = R(s, \pi(s)) + \gamma \sum_{s' \in S} p(s'|s,\pi(s)) V^{\pi}(s')$$

Let $x^\pi = [V^\pi(s)]_{s \in S}$. Then we wish to solve:

$$x^\pi = r^\pi + \gamma P^\pi x^\pi \Longleftrightarrow x^\pi = (I - \gamma P^{\pi})^{-1} r^{\pi}$$

where we used $$r^{\pi} = [R(s, \pi(s))]_{s \in S}$$, and $$P^{\pi} = [p(s'\|s,\pi(s))]_{s,s'\in S}$$.

Note that, because $0 \leq \gamma < 1$, the maxtrix $I - \gamma P^{\pi}$ is invertible (because the maximum eigenvalue of $P^{\pi}$ is $1$, because it is a transition probability matrix).

Does policy iteration converge? It can be verified that each iteration improves the value function until convergence; that is:

$$V^{\pi^{k+1}}(s) \geq V^{\pi^k}(s)$$ 

Note that the total number of policies is $\|A\|^{\|S\|}$. Therefore, policy iteration must converge in at most $\|A\|^{\|S\|}$ steps! On the other hand, in value iteration, we typically never converge to the exact result.

Therefore, in value iteration each iteration is cheaper ($O(\|S\|^2\|A\|)$) but it takes more iterations, while in policy iteration each iteration is expensive (we need to invert a matrix of dimensions $\|S\| \times \|S\|$) but it takes a finite amount of iterations.

### State-action value function Q Function

Sometimes Q functions are used instead of value functions. A Q function
is a function $Q^\pi: S \times A \to \mathbb{R}$ for a policy
$\pi: S \to A$, defined as:

$$Q^{\pi}(s,a) = R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a)V^{\pi}(s')$$

This corresponds to the expected reward if taking action $a$ at step
$s$. The optimal Q function satisfies:
$$Q^{*}(s,a) = R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a)V^{\pi^*}(s')$$
Hence, the optimal policy is:
$$\pi^*(s) = \mathop{\mathrm{argmax}}_{a \in A} Q^{*}(s, a)$$

### Optimality through Linear Programming

We know that the optimal value function satisfies: $$\begin{aligned}
V^{\pi^{*}}(s) &= \max_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi^{*}}(s') \right]\\
&\geq R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V^{\pi^{*}}(s'), \quad \forall a \in A
\end{aligned}$$

This result suggests creating the following linear program, where we
optimize over the value function $V : S \to \mathbb{R}$:

$$\begin{aligned}
\operatorname{minimize} & \quad \sum_{s \in S} V(s)\\
\text{such that} & \quad V(s) \geq R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V(s'), \quad \forall s \in S, \forall a \in A
\end{aligned}$$ 

We will argue that the optimal solution is $$V^{\pi^*}$$. It is clear that $$V^{\pi^*}$$ is a feasible solution to the linear program. To see that it is optimal, consider any other feasible $V$. Because $V \neq V^{\pi^*}$ and the optimal value function is unique, there must exist a state $s$ such that:

$$V(s) > R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V(s'), \quad \forall a \in A$$

But then it is possible to decrease the objective of the linear program by setting:

$$V(s) = \max_{a\in A} \left[ R(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) V(s') \right]$$

It is not difficult to verify that this solution is still feasible. Therefore, the optimal solution to the linear program must be $V^{\pi^*}$.

The linear programming formulation is useful because one can then formulate the dual linear program, and use it to obtain certificates of optimality.

## Model Predictive Control (MPC)

Model predictive control is a generic method that refers to approximately solving for the optimal policy. It is also known as *receding horizon control* or *rolling horizon planning*. Classically, it is useful in the context of linear dynamical system models.

Suppose we are only interested in a planning horizon $t$. That is, we are not interested in rewards that are obtained after more than $t$ steps. Note that this lies in-between the case $\gamma=0$ (where we are only interested in the immediate reward) and the case $\gamma = 1$ (where we are interested in all rewards equally). 

We will use the Q function to explain model predictive control. Using $s_0 = s$, we set:

$$Q(s_0, a_0) = R(s_0, a_0) + \gamma \sum_{s_1 \in S} p(s_1|s_0,a_0) \max_{a_1 \in A} Q(s_1, a_1)$$

$$Q(s_1, a_1) = R(s_1, a_1) + \gamma \sum_{s_2 \in S} p(s_2|s_1,a_1) \max_{a_2 \in A} Q(s_2, a_2), \quad \forall s_1 \in S(s_0, 1), a_1 \in A$$

$$\vdots$$

$$Q(s_{t-1}, a_{t-1}) = R(s_{t-1}, a_{t-1}) + \gamma \sum_{s_{t} \in S} p(s_t|s_{t-1},a_{t-1}) \max_{a_t \in A} Q(s_t, a_t), \quad \forall s_{t-1} \in S(s_0, t-1), a_{t-1} \in A$$

$$Q(s_t, a_t) = 0, \forall s_t \in S(s_0, t), a_{t} \in A$$

where $S(s, t)$ are the states that are reachable starting from $s$ in $t$ steps. Th equations show that, after $t$ steps, we consider all the rewards to b $0$. Because of this shifting horizon, the evaluation of the Q function needs to be repeated after every step. However, at every step, it may b only necessary to solve over a small subspace of the entire domain of Q because we may have $\|S(s, t)\|$ much smaller than $\|S\|$. This makes th method potentially faster than value iteration.

## Linear Quadratic Regulator 

The linear quadratic regulator is a special case of a decision making problem. We define the following program:

$$\begin{aligned}
\operatorname{minimize} & \quad J = \sum_{t=0}^\infty l(x(t), u(t))\\
\text{such that} &\quad u(t) \in \mathcal{U}, \quad x(t) \in \mathcal{X}, \quad t=0,1,...\\
&\quad x(t+1) = A x(t) + B u(t), \quad t=0,1,...\\
&\quad x(0) = z
\end{aligned}$$

Intuitively, the goal of the problem is to select the input trajectories such as to minimize the loss. 

The variable of the program are the states $x(0), x(1), ... \in \mathbb{R}^n$ and the input trajectories $u(0), u(1), ... \in \mathbb{R}^m$. The problem data (i.e. what is given) is: 
- The dynamics matrix $A\in \mathbb{R}^{n \times n}$ and the input matrix $B \in \mathbb{R}^{n \times m}$.

- The convex cost function $l : \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}$, with
  $l(0,0)=0$.

- The convex state constraint set $\mathcal{X}$ and the convex input constraint set $\mathcal{U}$, with $0 \in \mathcal{X}$, $0 \in \mathcal{U}$.

- The initial state $z \in \mathcal{X}$.

A special case, called the linear-quadratic problem, is given by setting:

- $\mathcal{X} = \mathbb{R}^n$, $\mathcal{U} = \mathbb{R}^m$.

- $l(x(t), y(t)) = x(t)^T Q x(t) + u(t)^T R u(t)$, with $Q \succeq 0$,
  $R \succ 0$.

This special case can be solved using dynamic programming. It turns out that the value function is quadratic: $$V(z) = z^T P z$$ In turn, $P$ can be found by solving an algebraic Riccati equation (ARE):

$$P = Q + A^TPA - A^T P B (R + B^T PB)^{-1} B^T P A$$ 

Then, the optimal policy consists of linear state feedback: $u^*(t) = K x(t)$, where $K = -(R + B^T P B)^{-1} B^T P A$.

Note that this implies that the state at time $t$, under the optimal
policy, is: $$x(t) = (A+BK) x(t-1) \Longrightarrow x(t) = (A+BK)^{t}z$$

<!-- todo define the value function here  -->
- Value is a long-term thing, reward is a short-term
- Value is not a number, but a function

# Value functions definitions
## The V-values 
## The Q-values
## The optimal V values
## The optimal Q values

# From definition to Bellman recursion

# Value-based methods

## Value iteration

## Policy iteration

## Q-learning