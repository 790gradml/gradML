---
title: "Lectures"
nav_order: 2
nav_exclude: true
---

# Lectures

Lecture recordings are uploaded to [Panopto](https://canvas.mit.edu/courses/33927/external_tools/594).

{% assign limit_value = 24 %}  <!-- Set this to the number of lectures to display-->
{% assign sorted_lectures = site.lectures | sort: 'id' %}
{% assign filtered_lectures = sorted_lectures | slice: 0, limit_value %}

{% for lecture in filtered_lectures %}
## {{ lecture.title }}
- {% if lecture.notes %} **[Notes]({{ lecture.notes }})** {% else %} **Notes:** *To be released* {% endif %}
- {% if lecture.slides %} **[Slides]({{ lecture.slides }})** {% else %} **Slides:** *To be released* {% endif %}
{% if lecture.additional_links %} - **Additional Links:** {% for link in lecture.additional_links %} 
    - [{{ link.name }}]({{ link.link }}) {% endfor %} {% endif %}
{% endfor %}


<!-- - {% if lecture.recording %} **[Recording]({{ lecture.recording }})** {% else %} **Recording:** *To be released* {% endif %} -->