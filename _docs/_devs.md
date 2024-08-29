## Developer Setup

All commands below should be run as command-line shell commands. For example, on Mac, these commands can be run in the built-in App `Terminal`, or `iterm2` (a popular 3rd-party alternative).

1. Clone this repository: `git clone https://github.com/790gradml/gradML.git`

2. (Important) `cd` to the directory of your clone of this repo

3. Install [Ruby](https://www.ruby-lang.org/en/)
    - If on Mac (tested on a fresh-installed MacOS 13.6):
        
        - Install `homebrew`
        - `brew install chruby`
        - `brew install rbenv ruby-build`
        - `rbenv install 3.1.3`
        - add `eval "$(rbenv init -)"` to your shell, e.g. to `.zshrc`
        - `rbenv local 3.1.3`
    
    - If on other OS, see https://github.com/790gradml/gradML/pull/11 for some details

4. Install [Jekyll](https://jekyllrb.com): `gem install bundler jekyll`

5. Install all this repo's specific dependencies `bundle install` 

6. Get the site running locally via `bundle exec jekyll serve`

7. Open your browser, and go to http://localhost:4000; you should see something that looks familiar 🥳