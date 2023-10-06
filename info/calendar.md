---
title: Calendar
description: Listing of course modules and topics.
nav_order: 0
#nav_exclude: true

---

# Calendar

{: .warning}

Rough semester calendar draft; dates/events are subject to change/update.
{% for module in site.modules %}
{{ module }}
{% endfor %}


# Recommended Reading

All freely accessible (an MIT IP may be required):

- [B] [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), Bishop; Springer, 2006.
- [HTF] [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), Hastie, Tibshirani, and Friedman; Springer, 2009.
- [SB]/[SSS] [Understanding Machine Learning: From Theory to Algorithms](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning), Shalev-Shwartz and Ben-David; Cambridge University Press, 2014.
- [SB] [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020trimmed.pdf), Sutton and Barton; The MIT Press, 2018.
- [JWHT] [An Introduction to Statistical Learning](https://www.statlearning.com/), James, Witten, Hastie, and Tibshirani; Springer, 2013.
<!-- - [BV] [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), Boyd and Vandenberghe; Cambridge University Press, 2004 -->
