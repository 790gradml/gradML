---
title: "Homeworks"
nav_order: 1
nav_exclude: true
---

# Homeworks

Submit homeworks to [gradescope](https://www.gradescope.com/courses/1112729).

{% assign limit_value = 8 %}  <!-- Set this to the number of hws to display-->
{% assign sorted_homeworks = site.homeworks | sort: 'due_date' %}
{% assign filtered_homeworks = sorted_homeworks | slice: 0, limit_value %}

<!-- Debugging output -->
<!-- <p>Limit Value: {{ limit_value }}</p>
<p>Total Items: {{ total_items }}</p>
<p>Start Index: {{ start_index }}</p> -->
<!-- <p>Sorted Homeworks:</p>
<pre>{{ sorted_homeworks | inspect }}</pre> -->
<!-- <p>Filtered Homeworks:</p>
<pre>{{ filtered_homeworks |inspect }}</pre> -->

{% for hw in filtered_homeworks %}
## {{ hw.title }}

<!-- - **Release Date:** {{ hw.release_date | date: "%B %d, %Y" }} -->
- **Due Date:** {{ hw.due_date | date: "%B %d, %Y" }}
- {% if hw.pdf %} **[PDF]({{ hw.pdf }})** {% else %} **PDF:** *To be released* {% endif %}
- {% if hw.sln %} **[Solution]({{ hw.sln }})** {% else %} **Solution:** *To be released* {% endif %}
{% if hw.additional_links %} - **Additional Links:** {% for link in hw.additional_links %} 
    - [{{ link.name }}]({{ link.link }}) {% endfor %} {% endif %}

{% endfor %}

<!-- - {% if hw.gradescope_link %} **[Submit to Gradescope]({{ hw.gradescope_link }})** {% else %} **Submit to Gradescope:** *To be released* {% endif %} -->
