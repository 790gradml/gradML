## Minor edits and "Live" site
Minor edits like typo fix can be done directly by following the edit link at the bottom of each page. 

Those with this repo's write access can edit directly via the web UI; or, PR can be another approach. Any push to the `main` branch will be "live" on `gradML` site. 

Below is a quick demo of adding a "gradient vector" blob to the site:
![](figs/simpe-edit-demo.gif)


## Site structure 

The website structure largely mirrors the file system structure (with a few exceptions, detailed later).

Firstly, "top-level" navigational pages are placed under the root folder, for example, the site renders the [https://gradml.mit.edu/review/](https://gradml.mit.edu/review/) on the homepage since the corresponding source [review.md](https://github.com/shensquared/gradML/blob/main/review.md) lives under the repo root.

Lower level structure is declared in the [`_config.yml`](https://github.com/shensquared/gradML/blob/main/_config.yml) file. For example, blobs like 
```
- scope:
          path: "reinforcement/"
      values:
          parent: Reinforcement Learning
          layout: page
``` 
in the [`_config.yml`](https://github.com/shensquared/gradML/blob/main/_config.yml) is how the site knows to look for markdown files located under the `<repo root>/reinformcement` folder to render the corresponding drop-down module. (Similarly, under `<repo root>/supervised/` we have a `learnability_and_vc.md` file; this file is the source for rendering the [https://gradml.mit.edu/supervised/learnability_and_vc/](https://gradml.mit.edu/supervised/learnability_and_vc/) page.)


Markdown files with filenames starting with `_` or located under a folder whose name starts with `_` are skipped. E.g. no files under the `_docs` folder are rendered. 

