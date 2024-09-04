---
# layout: default
title: "Homeworks"
nav_order: 6
# permalink: /homeworks/
---

# Homeworks

{% assign sorted_homeworks = site.homeworks | sort: 'release_date' | reverse %}

{% for hw in sorted_homeworks %}
## {{ hw.title }}

- **Release Date:** {{ hw.release_date | date: "%B %d, %Y" }}
- **Due Date:** {{ hw.due_date | date: "%B %d, %Y" }}
- **[Download PDF]({{ hw.pdf }})**
- **[Submit to Gradescope]({{ hw.gradescope_link }})**

{% endfor %}
