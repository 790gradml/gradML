---
title: Markov Decision Process
nav_order: 3
---



# MDP

## Outline

Today's lecture will come in three parts. We'll first formally introduce a mathematical framework called Markov Decision Processes, and in particular, a central concept there called the value functions.  We'll then introduce two dynamic-programming algorithms that solve for these value functions and thereby solve an MDP. And finally, if there's time, we will describe the leap that we take from MDP to RL, and the so-called Q learning algorithm.

So the objective: familiar with four definitions: MDP, RL, the two value functions.

## Recall: Bandits
We've seen [Bandits](/reinforcement/bandit/) problems are one type of problems that can fit in this looped diagram. You're presented with $k$ slot machines, collectively, those machines are your state. Your job is to make a decision as to which slot machine to pull over some period of time.

The bandits problem illustrates the fundamental tradeoff between exploitation and exploration. And the reason that there's this tradeoff is because we do not have access to the rewards function. The rewards are a stochastic function of the states, and we do not know the explicit form, or the parameters of the function.

A natural question arises, what if in the bandits case, we do know the rewards function form?

This would actually be the so-called stochastic optimization problem. Which is itself a pretty important type of decision-making problems. It can often be solved via supervised learning. We're not gonna discuss this setup in detail. But rather, as we laid out A good exercise could be, after today's lecture, come back to this table, and perhaps dream up with situations where you can model the problem as a stochastic optimization problem. 

But for today, we're gonna first take things in another direction.

## MDP Definitions
A markov decision process describes a problem in which an agent
interacts with the environment: at each step, the agent takes an action,
to which the environment responds by giving a reward and by changing the
state of the agent.

Formally, a markov decision process is a tuple $(S, A, p, R, \gamma)$.


- $S$: a state space which contains all possible states of the system
- $A$: an action space which contains all possible actions an agent can take
- $P(s'\|s,a)$ a stochastic transition kernel which gives the probability of transition from state $s$ to $s'$ if action $a$ is taken. This transition can be deterministic as well. Implies by this notation is that $s$ is the current state; $s'$ is the next state, or the state we ended in.
- $R: S \times A \to \mathbb{R}$: a reward function that takes in the state and the action and gives back a real value reward.
- $\gamma \in [0,1]$: a discount factor that re-weight long term and short term rewards

The policy we want to derive, denoted as $\pi$, is a function $S \to A$, which decides the action given the state. 

Finally, the objective of decision making is to find a policy $\pi$ that maximizes the cummulative reward for every starting state $s \in S$ across steps:

$$\mathbb{E} \left[\sum_{t=0}^{T}\gamma^t R(s_t,a_t)\Big|s_0=s\right]$$

## MDP general comments 
- MDP definition sometimes also has a horizon.
- Math is cleaner in infinite-horizon case. Why? well, because when things have been run for a long long time, things would have "settled down" or "converged" and the past history -- at whatever timestamp you slice it -- no longer matter.  In the literature, sometimes finite-horizon problems are also called episodic; whereas infinite-horizon ones are called continuing task.

As an aside, this is a general pattern. Asymptotic regime allows us to "cancel out" a lot of things, take limits, etc. (But not very practical; this intrinsic tension motivates lots of research in engineering)

A couple of more comments:
- States versus observation. See POMDP. 
- Rewards are what we immediately see but not our final goal. We're in the game for the long run, so it's the sum of rewards that we go after. These are the so-called value functions, which we'll define more precisely and need to comment with more details. For now, remember this intuitive punchline: rewards are short-term thing, values are long-term thing. We want to do good in the long run.
- Fundamental to the long-term versus short-term performance is the tension between exploitation and exploration.
Exploitation does well in the short-run; exploration may help us do well in the long-run (even if potentially losing out somewhat in the short-term).
- Sometimes in the literature, the rewards model would be a mapping of (s, s', a) to (r). In other words, the reward you get not only depends on your current state, and your current action, but also the state you ends in (based on the generally stochastic transition). One such example would be, when you buy a lottery at a particular store, the (monetary) return/reward you get depends on if you actually win. Note, however, that such reward model distinction really do not matter at all in the infinite-horizon case -- again, coz convergence cancels that out. Even in the finite-horizon case, all the math works the same by thinking of the future-dependent reward as future-independent, but the horizon got shortened by 1.
- In the infinite-horizon set up, the discount factor has to be strictly less than 1, for making sure the infinite-length rewards sequence converge and we end up with a finite summation of infinite terms.

## POMDP



