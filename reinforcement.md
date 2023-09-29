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
# Sequential Decision-making Problems
- The four big types of sequential decision-making problems
    - Bandits
    - MDP
    - RL
    - Stochastic optimization

and their connections and differences

# MDP

- State
- Action
- Reward
- Transition
- Discount factor

Sometimes also horizon. 

Math is cleaner in infinite-horizon case. Why? well, because when things have been run for a long long time, things would have "settled down" or "converged" and the past history -- at whatever timestamp you slice it -- no longer matter.

As an aside, this is a general pattern. Asymptotic regime allows you to "cancel" out a lot of things, take limits, etc. (But not very practical; this intrinsic tension motivates lots of research in engineering)
# Reinforcement Learning Set Up

No longer have access to the transition model 

