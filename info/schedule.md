---
title: Weekly Schedule
description: The weekly event schedule.
nav_order: 2
---

# Weekly Schedule

Some quick notes:
- TA office hours are subject to minor changes.
- TA office hours locations are virtual throughout the first week except of the Friday office hours. We will update the location information here as soon as the registrar confirms the room assignments for the semester.
- In a nominal week, we have two lectures (each covers different material). In addition, we encourage you to attend the office hours to ask any questions about the course material.
- Assignment deadlines don't appear on the weekly schedule. Please check the [course calendar](https://gradml.mit.edu/info/calendar/) to view the deadlines.

{% for schedule in site.schedules %}
{{ schedule }}
{% endfor %}
