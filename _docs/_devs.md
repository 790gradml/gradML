## Developer Setup
(very much WIP...)

All commands below should be run as command-line shell commands. (e.g. on Mac, you run these in the build-in App `Termial`, or similarly in `iterm2`)

1. Clone this repository: `git clone https://github.com/shensquared/gradML.git`
2. `cd` to the directory of your clone of this repo.
3. Install [Ruby](https://www.ruby-lang.org/en/) if it's not already installed (see https://github.com/shensquared/gradML/pull/11 for some details)
4. Install `Jekyll`(https://jekyllrb.com): `gem install bundler jekyll`
5. Install all this repo-specific dependencies `bundle install` (important you're at the root directory of your gradML clone)
6. Get the site running locally via running `bundle exec jekyll serve`
7. Open your browser and go to http://localhost:4000 and you should see something that looks familiar 🥳