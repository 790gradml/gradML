# Site structure 

- First, a few "top-level" navigational pages are declared in the [`_config.yml`](https://github.com/shensquared/gradML/blob/main/_config.yml) file. For example, the site renders [https://gradml.mit.edu/review/](https://gradml.mit.edu/review/) page 

- The website structure largely mirrors file system structure (with a few exceptions, detailed later). 
- For example, under `<repo root>/supervised/` we have a `learnability_and_vc.md` file; this file is the source for rendering the [https://gradml.mit.edu/supervised/learnability_and_vc/](https://gradml.mit.edu/supervised/learnability_and_vc/) page.

- Markdown files with filenames starting with `_` or located under a folder whose name starts with `_` are skipped. E.g. no files under the `_docs` folder are rendered. 

## Edits and "Live" site
- Minor edits like typo fix can be done directly on GitHub. Those with repo-write access can edit directly via the web UI; PR can be another approach. Any push to this repo's `main` branch will be "live" on `gradML` site. 

![](figs/simpe-edit-demo.gif)