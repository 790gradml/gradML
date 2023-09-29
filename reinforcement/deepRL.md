---
title: Deep Reinforcement Learning
nav_order: 5
---
Deep RL = use deep learnin tools to solve traditional RL problems. In some sense, it can be simply thought of as snapping a deep neural network onto whatever we use to learn -- similar to the leap from logistic regression for basic classication, to the modern day deep nets for classificaton. But benethe this also lies all sorts of new challenges, solutions, and insights -- after all, deep learning is itself a sort of standalone subject (and we do have a class on that at EECS; shout out to Philip!)


- Considerations when choosing an algo
    - Sample complexity
    - The use of a model
        
        (emphasize the model here differs from the hypothesis model; in RL and control context, model typically refers to the dynamics model. the usual ML hypothesis or model is typically a policy, or a value function)
    <!-- - Whether the dynamics model is super discrete -->

# TROP, PPO