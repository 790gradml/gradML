---
layout: about
title: Reinforcement Learning
has_children: true
nav_order: 6
has_toc: false
---

{: .no_toc}
## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

# Overview
# Sequential Decision-making Problems (Under Uncertainty)
- The four big types of sequential decision-making problems
    - Bandits
    - MDP
    - RL
    - Stochastic optimization

and their connections and differences. The "under uncertainty" modifier is really what makes these problems interesting and practical.

## Different fields that study these problems:

- Control theory (especially robust control)
- Operations research
- Finance
- Social sciences

Some comments on each field's unique challenges. On one hand, good that a single framework has so many applications; on the other hand though, inevitable notational mess.


# MDP

- $S$: a state space which contains all possible states of the system
- $A$: an action space which contains all possible actions an agent can take
- $P(s'\|s,a)$ a stochastic transition kernel which gives the probability of transition from state $s$ to $s'$ if action $a$ is taken. This transition can be deterministic as well. Implies by this notation is that $s$ is the current state; $s'$ is the next state, or the state we ended in.
- $R: S \times A \to \mathbb{R}$: a reward function that takes in the state and the action and gives back a real value reward.
- $\gamma \in [0,1]$: a discount factor that re-weight long term and short term rewards

Sometimes also horizon. 

Math is cleaner in infinite-horizon case. Why? well, because when things have been run for a long long time, things would have "settled down" or "converged" and the past history -- at whatever timestamp you slice it -- no longer matter.  In the literature, sometimes finite-horizon problems are also called episodic; whereas infinite-horizon ones are called continuing task.

As an aside, this is a general pattern. Asymptotic regime allows you to "cancel" out a lot of things, take limits, etc. (But not very practical; this intrinsic tension motivates lots of research in engineering)

A couple of more comments:
- States versus observation. 
- Rewards are what we immdiately see but not our final goal. We're in the game for the long run, so it's the sum of rewards that we go after. These are the so-called value functions, which we'll define more precisely and need to comment with more details. For now, remember this intuitive punchline: rewards are short-term thing, values are long-term thing. We want to do good in the long run.
- Fundamental to the long-term versus short-term performance is the tension between exploitation and exploration.
Exploitation does well in the short-run; exploration may help us do well in the long-run (even if potentially losing out somewhat in the short-term).
- Sometimes in the literature, the rewards model would be a mapping of (s, s', a) to (r). In other words, the reward you get not only depends on your current state, and your current action, but also the state you ends in (based on the generally stochastic transition). One such example would be, when you buy a lottery at a particular store, the (moneitary) return/reward you get depends on if you actually win. Note, however, that such reward model distinction really do not matter at all in the infinite-horizon case -- again, coz converegence cancels that out. Even in the finite-horizon case, all the math works the same by thinking of the future-dependent reward as future-independent, but the horizon got shortened by 1.
- In the infinite-horizon set up, the discount factor has to be strictly less than 1, for making sure the infinite-length rewards sequence converge and we end up with a finite summation of infinite terms.

# Reinforcement Learning Set Up

## Agent and enviroment


No longer have access to the transition model, and sometimes no access to the rewards model either. 

So there're the loss of two key components here; losing either makes the problem hard. Different applications might have different levels of lost access. For instance, often times in rigid-body robotics, we have at least the law of physics as a guiding principal and from that we get vague ideas of the transition.



