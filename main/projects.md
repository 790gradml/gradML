---
title: "Projects"
nav_order: 3
nav_exclude: true
---

# Projects

Submit projects to [gradescope](https://www.gradescope.com/courses/1112729).

{% assign limit_value = 2 %}  <!-- Set this to the number of projects to display-->
{% assign sorted_projects = site.projects | sort: 'due_date' %}
{% assign filtered_projects = sorted_projects | slice: 0, limit_value %}

<!-- Debugging output -->
<!-- <p>Limit Value: {{ limit_value }}</p>
<p>Total Items: {{ total_items }}</p>
<p>Start Index: {{ start_index }}</p> -->
<!-- <p>Sorted projects:</p>
<pre>{{ sorted_projects | inspect }}</pre> -->
<!-- <p>Filtered projects:</p>
<pre>{{ filtered_projects |inspect }}</pre> -->

{% for proj in filtered_projects %}
## {{ proj.title }}

<!-- - **Release Date:** {{ proj.release_date | date: "%B %d, %Y" }} -->
- **Due Date:** {{ proj.due_date | date: "%B %d, %Y" }}
- {% if proj.pdf %} **[PDF]({{ proj.pdf }})** {% else %} **PDF:** *To be released* {% endif %}
- {% if proj.sln %} **[Solution]({{ proj.sln }})** {% else %} **Solution:** *To be released* {% endif %}
{% if proj.description %} - {{proj.description}} {% endif %}
{% if proj.additional_links %} - **Additional Links:** {% for link in proj.additional_links %} 
    - [{{ link.name }}]({{ link.link }}) {% endfor %} {% endif %}

{% endfor %}

<!-- - {% if proj.gradescope_link %} **[Submit to Gradescope]({{ proj.gradescope_link }})** {% else %} **Submit to Gradescope:** *To be released* {% endif %} -->