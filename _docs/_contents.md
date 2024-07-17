## Minor edits and "Live" site
Minor edits like typo fix can be done by following the `edit link` at the bottom of each `gradML` webpage. 

Those with this repo's write access can edit directly via the GitHub webUI; or, PR can be another approach. Content-edit pushed to the `main` branch will be "live" on `gradML` within a few second (in contrast, other changes like website infastructure upgrade or feature change requires a server-side reload).

Below is a quick demo of adding a "gradient vector" blob:
![](figs/simpe-edit-demo.gif)


## Site structure 

Overall, the website structure mirrors the file system structure (with a few exceptions, detailed later). Every `html` webpage is sourced from a `.md` markdown file. Any markdown file with filename starting with `_` or located under a folder whose name starts with `_` are skipped and not turned into a webpage, e.g. no files under this `_docs` folder are rendered on `gradML`.

Top-level navigational pages are placed under the root folder, for example, the site links the [https://gradml.mit.edu/review/](https://gradml.mit.edu/review/) on the homepage since the corresponding source [review.md](https://github.com/shensquared/gradML/blob/main/review.md) lives under the repo root folder.

Lower-level structure is declared in the [`_config.yml`](https://github.com/shensquared/gradML/blob/main/_config.yml) file. For example, this blob
```
- scope:
    path: "reinforcement/"
  values:
    parent: Reinforcement Learning
    layout: page
``` 
in the [`_config.yml`](https://github.com/shensquared/gradML/blob/37564ca75c73b216f16c1ef165721417ab78ed6b/_config.yml#L144) is how the site knows to look for markdown files located under the `<repo root>/reinformcement` folder to orangize the corresponding drop-down module. (Similarly, under `<repo root>/supervised/` we have a `learnability_and_vc.md` file; this file is being rendered to the [https://gradml.mit.edu/supervised/learnability_and_vc/](https://gradml.mit.edu/supervised/learnability_and_vc/) page.)





