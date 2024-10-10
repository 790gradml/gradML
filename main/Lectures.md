---
title: "Lectures"
nav_order: 2
---

# Lectures

Lecture recordings are uploaded to [Panopto](https://mit.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx#folderID=%22b2079bc7-0ca3-4fdf-a3e5-b1d4014c37ee%22).

{% assign limit_value = 11 %}  <!-- Set this to the number of lectures to display-->
{% assign sorted_lectures = site.lectures | sort: 'id' %}
{% assign filtered_lectures = sorted_lectures | slice: 0, limit_value %}

{% for lecture in filtered_lectures %}
## {{ lecture.title }}
- {% if lecture.notes %} **[Notes]({{ lecture.notes }})** {% else %} **Notes:** *To be released* {% endif %}
- {% if lecture.slides %} **[Slides]({{ lecture.slides }})** {% else %} **Slides:** *To be released* {% endif %}

{% endfor %}


<!-- - {% if lecture.recording %} **[Recording]({{ lecture.recording }})** {% else %} **Recording:** *To be released* {% endif %} -->