---
title: Weekly Schedule
description: The weekly event schedule.
nav_order: 2
---

# Weekly Schedule

Some quick notes:
- TA office hours will be posted soon.
- In a nominal week, we have two lectures (each covers different material) and four recitation sessions (all four cover the same material).

{% for schedule in site.schedules %}
{{ schedule }}
{% endfor %}
