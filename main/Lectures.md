---
title: "Lectures"
nav_order: 2
---

# Lectures

Find notes from lectures below.

{% assign limit_value = 2 %}  <!-- Set this to the number of lectures to display-->
{% assign sorted_lectures = site.lectures | sort: 'id' %}
{% assign filtered_lectures = sorted_lectures | slice: 0, limit_value %}

{% for lecture in filtered_lectures %}
## {{ lecture.title }}
- {% if lecture.notes %} **[Notes]({{ lecture.notes }})** {% else %} **Notes:** *To be released* {% endif %}
- {% if lecture.slides %} **[Slides]({{ lecture.slides }})** {% else %} **Slides:** *To be released* {% endif %}
- {% if lecture.recording %} **[Recording]({{ lecture.recording }})** {% else %} **Recording:** *To be released* {% endif %}

{% endfor %}
