---
title: "Lectures"
nav_order: 2
---

# Lectures

Find notes from lectures below.

{% assign limit_value = 1 %}  <!-- Set this to the number of lectures to display-->
{% assign sorted_lectures = site.lectures | sort: 'id' | reverse %}
{% assign total_items = sorted_lectures | size %}
{% assign start_index = total_items | minus: limit_value %}
{% assign filtered_lectures = sorted_lectures | slice: start_index, limit_value %}

{% for lecture in filtered_lectures %}
## {{ lecture.title }}
- {% if lecture.notes %} **[Notes]({{ lecture.notes }})** {% else %} **Notes:** *To be released* {% endif %}
- {% if lecture.slides %} **[Slides]({{ lecture.slides }})** {% else %} **Slides:** *To be released* {% endif %}

{% endfor %}
