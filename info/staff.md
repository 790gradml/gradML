---
title: Staff
description: A listing of all the course staff members.
nav_order: 1
in_row: true
#nav_exclude: true
---


# Instructors
{% if page.in_row %}
<div class="staff-row">
{% else %}
<div>
{% endif %}
{% assign instructors = site.staffers | where: 'role', 'Instructor' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}
</div>


{% assign teaching_assistants = site.staffers | where: 'role', 'Teaching Assistant' %}
{% assign num_teaching_assistants = teaching_assistants | size %}
{% if num_teaching_assistants != 0 %}

# Teaching Assistants
{% if page.in_row %}
<div class="staff-row">
{% else %}
<div>
{% endif %}
{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}
</div>
{% endif %}