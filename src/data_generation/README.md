# Prerequisites

* Install numpy, nltk, BeautifulSoup 

# NOTE

Data for the sites: askubuntu.com, unix.stackexchange.com & superuser.com is already included in this repo
The original Stack Exchange data dump used to generate this dataset can be found here: https://archive.org/download/stackexchange

# Steps to generate data for any other site

1. Choose a sitename from the list of sites in https://archive.org/download/stackexchange. Let's say you chose 'academia.com'
2. Download the .7z file corresponding to the site i.e. academia.com.7z and unzip it under ranking_clarification_questions/stackexchange/
3. Set "SITENAME=academia.com" in ranking_clarification_questions/src/data_generation/run_data_generator.sh
4. cd ranking_clarification_questions; sh src/data_generation/run_data_generator.sh

This will create data/academia.com/post_data.tsv & data/academia.com/qa_data.tsv files
