# South Sudan Project 

HOW TO RUN:
1. Install requirements.txt
2. In your environment, execute <python -m spacy download en_core_web_sm>
3. Run topic_modelling.ipynb
4. Run topic_modelling_LDA.ipynb
5. Run predictions.ipynb

## Code description

You are provided with three notebooks. The notebooks depend on the helper function file for large functions. Also given are four .csv datasets and a zip file containing a .csv dataset.

In the topic_modelling.ipynb notebook the zipped dataset merged_summaries_fullarticles.zip is unzipped. The zip file contains a dataset which is incorrectly joined and has created nearly 300.000 faulty datapoints. By comparing the unzipped file to articles_summary_cleaned.csv we can drop all joins that were done incorrectly. When this is done a BERTopic model is fit on a subset of corrected_full_dataset which only contains the summaries. Then, keyword extraction by means of Yake and Rake methods are applied on the full paragraphs and the summaries of corrected_full_dataset resulting in 4 keyword lists. These categories are used for categorising articles (or none if the article doesn’t match any of the categories) and thus creating features from the news articles. Finally visualizations for the overlap of rake/yake and pre/post 2015, and finally pre/post 2017 are made. The returned keywords were hardcoded so the visualizations can be created without actually having the create the models, and the date split models are not used in further code. Do not run past "Modelling 2015-2017" if only interested in running predictions.ipynb on the unsplit dataset, as the code below this note takes a while to run (1+ hour runtime on HP ZBook Studio G5 I7-9750h).

In topic_modelling_LDA.ipynb, the topic modelling and keyword extraction method LDA is used to generate topics. The notebook may take a long time to run.

In predictions.ipynb an exploration of missing data is created along with fitting an OLS regression and a logistic regresion model on the data, with- and without the a few base features. The model is fit on an increasing number of topics to explore the effect of the number of topics. The notebook uses the food_crises_cleaned.csv dataset and the csv files obtained from the  topic_modelling.ipynb and topic_modelling_LDA.ipynb notebooks. Finally, visualisations are made to explore the effects of the number of topics and the different models on a number of different metrics.


## Requirements
To install the requirements open Terminal (macOS)/Command Prompt (Windows) and run pip install -r requirements.txt. If you create a new environment in PyCharm, an icon should appear to install requirements. The code runs with Python 3.10.12. It is highly advisable to use a new enviroment for each different project you open, as previously installed packages can have conflicting dependencies which can prevent code from running.

Required libraries:
- bertopic == 0.15.0
- pandas == 1.4.4
- geopandas == 0.13.2
- matplotlib == 3.7.2
- seaborn == 0.12.2
- statsmodels == 0.14.0
- rake-nltk == 1.0.4
- yake == 0.4.8
- ipywidgets == 8.1.1
- matplotlib_venn == 0.11.9

## Troubleshooting

If you encounter any issues while running the notebooks, try the following:
- check that you have all the necessary libraries installed and the correct versions of them
- check your Python version. In principle, the code should work with any Python versions higher than 3.10.12 If this is not the case, create a virtual environment that uses Python 3.10.12
- a previously encountered error is a dependency conflict involving bertopic and numpy. Make sure to not install libraries such as xgboost which depend on newer versions of numpy. 

## ChatGPT
The following Chatgpt interaction was used to gain the untrained gpt keywords:

<i>Consider the situation of South Sudan. Present 20 singular keywords in a list format that describe the past situation of South Sudan.</i>

Certainly, here are 20 keywords that describe the past situation of South Sudan:

<i>The keywords may only consist of a single word. abbreviations of parties are allowed</i>
Understood, here are 20 single-word keywords and abbreviations that describe the past situation of South Sudan:

<i>Civil War is not a single word, reconsider the list</i>
Certainly, here's a revised list of single-word keywords and abbreviations that describe the past situation of South Sudan:

<i>Can you put quotations around the words and a comma in between like a python list format?</i>
Certainly, here's the list of single-word keywords and abbreviations for the past situation of South Sudan in Python list format:

~