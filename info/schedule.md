---
title: Weekly Schedule
description: The weekly event schedule.
nav_order: 2
info_cat: true
---

# Weekly Schedule

Some quick notes:
- TA office hours will be posted soon.
- In a nominal week, we have two lectures (each covers different material). In addition, we encourage you to attend the office hours to ask any questions about the course material.

{% for schedule in site.schedules %}
{{ schedule }}
{% endfor %}
