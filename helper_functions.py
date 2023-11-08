
from bertopic import BERTopic
from umap import UMAP
import pandas as pd
import os
from rake_nltk import Rake
from tqdm.notebook import tqdm
import nltk
import yake
from ipywidgets import FloatProgress
from collections import Counter
import itertools
import ast
import re
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics

# Import models
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm


def yake_keyword(dataframe):
    """Applies the yake library to a dataframe. Yake is a library that applies keyword extraction.
    
    Input: 
    - dataframe: A dataframe consisting out of a column with text that needs keyword extraction

    Output:
    - dataframe: Dataframe containing 2 extra columns (paragraph & summary) with the keywords determined by yake.
    
    """
    # Implement a progress bar in the cell to show the progress.
    tqdm.pandas()
    # Apply the keyword extractor function from the NLP library yake.
    language = 'en'
    max_ngram_size = 2
    deduplication_threshold = 0.9
    numOfKeywords = 3  # Multiple keywords
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size,
                                                dedupLim=deduplication_threshold,
                                                top=numOfKeywords, features=None)
    extractor = lambda x: custom_kw_extractor.extract_keywords(x) # Extract the keywords with the yake KeywordExtractor.
    # Apply the keyword extraction on both the summaries and the whole article content
    dataframe['paragraphs_3_keywords_2gram_summary'] = dataframe['summary'].progress_apply(extractor)
    dataframe['keywords_paragraphs'] = dataframe['paragraphs'].progress_apply(extractor)
    return dataframe

# Function to extract words between '[' and ','
def extract_words_from_string(dataframe, column):
    """ Applies regex methods to clean the columns produced by the yake_keyword function.
    It removes the '[' and the ',' and the score behind the word.
    
    Input: Column of a dataframe you want to clean.
    
    Output: Updated (cleaned) column.
    """
    # Define a regular expression pattern
    pattern = r'\((.*?)\,'
    dataframe[column] = dataframe[column].str.extract(pattern, expand=False)

    return dataframe

def find_intersection(list1, list2):
    """Function to find the intersection between two lists
    Input: two lists containing strings"""
    set1 = set(list1)
    set2 = set(list2)
    return list(set1.intersection(set2))

def rake_extractor(text):
    """ Determines the keywords."""
    r = Rake()
    nltk.download('stopwords')
    nltk.download('punkt')
    r.extract_keywords_from_text(text)
    return list(r.get_word_degrees().keys())

def rake_keywords(dataframe):
    """Applies the rake library to a dataframe. Rake is a library that applies keyword extraction.
    Input: 
    - dataframe: A dataframe consisting out of a column with text that needs keyword extraction
    Output:
    - dataframe: Dataframe containing 2 extra columns (paragraph & summary) with the keywords determined by rake.
    """
    # Apply the extractor function with a progress bar to the 'summary' column
    tqdm.pandas(desc="Extracting Keywords from 'summary'")
    dataframe['summary_rake_keywords'] = dataframe['summary'].progress_apply(rake_extractor)

    # Apply the extractor function with a progress bar to the 'paragraphs' column
    tqdm.pandas(desc="Extracting Keywords from 'paragraphs'")
    dataframe['paragraphs_rake_keywords'] = dataframe['paragraphs'].progress_apply(rake_extractor)
    return dataframe

def get_relevant_topics(bertopic_model, keywords, top_n):
    '''
    # We create a function to calculate a list of the top n topics related to (a) given keyword(s)
    Retrieve a list of the top n number of relevant topics to the provided (list of) keyword(s)
    
    Parameters:
        bertopic_model: a (fitted) BERTopic model object
        
        keywords:   a string containing one or multiple keywords to match against,
                    
                    This can also be a list in the form of ['keyword(s)', keyword(s), ...]
                    
                    In this case a maximum of top_n topics will be found per list element 
                    and subsetted to the top_n most relevant topics.
                    
                    !!!
                    Take care that this method only considers the relevancy per inputted keyword(s) 
                    and not the relevancy to the combined list of keywords.
                    
                    In other words, topics that appear in the output might be significantly related to a 
                    particular element in the list of keywords but not so to any other element, 
                    
                    while topics that do not appear in the output might be significantly related to the 
                    combined list of keywords but not much to any of the keyword(s) in particular.
                    !!!
                    
        top_n: an integer indicating the number of desired relevant topics to be retrieved
        
        
        Return: a list of the top_n (or less) topics most relevant to the (list of) provided keyword(s)
    '''
    
    if type(keywords) is str: keywords = [keywords] # If a single string is provided convert it to list type
    
    relevant_topics = list() # Initilize an empty list of relevant topics
    
    for keyword in keywords: # Iterate through list of keywords
        
        # Find the top n number of topics related to the current keyword(s)
        topics = bertopic_model.find_topics(keyword, top_n = top_n)
        
        # Add the topics to the list of relevant topics in the form of (topic_id, relevancy)
        relevant_topics.extend(
            zip(topics[0], topics[1]) # topics[0] = topic_id, topics[1] = relevancy
        )
    
    
    relevant_topics.sort(key=lambda x: x[1]) # Sort the list of topics on ASCENDING ORDER of relevancy
    
    # Get a list of the set of unique topics (with greates relevancy in case of duplicate topics)
    relevant_topics = list(dict(relevant_topics).items())
    
    
    relevant_topics.sort(key=lambda x: x[1], reverse=True) # Now sort the list of topics on DESCENDING ORDER of relevancy
    
    return relevant_topics[:10] # Return a list of the top_n unique relevant topics

def add_keywords_as_columns_to_dataframe(chosen_words_list_all_possibilities,bertopic_model, dataframe):
    """Add the keywords generated by key word extraction method as a column to a dataframe 
    with a value containing True or False depending on wheteher the keyword relates to the article.

    Input:
    - chosen_words_list_all_possibilities: list of keywords

    Output:
    - dataframe
    """
    for item in tqdm(chosen_words_list_all_possibilities):
        # print(item)
        # Get the top 10 topics related to the keywords 'hunger' and 'food insecurity'
        relevant_topics = get_relevant_topics(bertopic_model = bertopic_model, keywords=item, top_n=10)

        topic_ids = [el[0] for el in relevant_topics] # Create seperate list of topic IDs

        # for topic_id, relevancy in relevant_topics: # Print neat list of (topic_id, relevancy) tuples
        #     print(topic_id, relevancy)

        item = str([item])   
        dataframe[item] = [t in topic_ids for t in bertopic_model.topics_] # Add boolean column to df if topic in list of relevant topics

        # View the Count, Name, Representation, and Representative Docs for the relevant topics
        # bertopic.get_topic_info().set_index('Topic').loc[topic_ids]
    return dataframe

def create_lag_df(df, columns, lag, difference=False, rolling=None, dropna=False):
    '''Here we define a function that lags input variables. 
    There are options for creating a rolling mean, taking the difference between subsequent rows, and dropping NaNs. 
    Feature engineering can of course be extended much further than this.

    Function to add lagged colums to dataframe
    
    Inputs:
        df - Dataframe
        columns - List of columns to create lags from
        lag - The number of timesteps (in months for the default data) to lag the variable by
        difference - Whether to take the difference between each observation as new column
        rolling - The size of the rolling mean window, input None type to not use a rolling variable
        dropna - Whether to drop NaN values
        
    Output:
        df - Dataframe with the lagged columns added
    '''
    
    for column in columns:
        col = df[column].unstack()
        if rolling:
            col = col.rolling(rolling).mean()
        if difference:
            col = col.diff()
        if dropna:
            col = col.dropna(how='any')
        df[f"{column}_lag_{lag}"] = col.shift(lag).stack()
    return df

def plot_ConfusionMatrix(prediction, true, binary=False):
    '''
    Function to plot a confusion matrix as a heatmap from a prediction and true values.
    Here we define a function that plots a confusion matrix given a prediction and the true values, it can be used both for binary and categorical variables.

    Inputs:
        prediction - The predicted values
        true - the true values
        binary - whether the variable is binary or not
        
    Output:
        confusion_matrix - The calculated confusion matrix based on the prediction and true values.
        
        Also plots the confusion matrix as heatmap in an interactive environment such as Jupyter Notebook.
    '''
    
    y_pred = prediction
    
    if not binary:
        # Round prediction to nearest integer (i.e. the nearest phase)
        y_pred = y_pred.round() 
        y_pred = np.minimum(y_pred, 5) # Cap maximum prediction at 5 (maximum phase)
        y_pred = np.maximum(y_pred, 1) # Cap minimum prediction at 1 (minimum phase)

    # Initialize confusion matrix
    confusion_matrix = pd.crosstab(
        true, y_pred, rownames=["Actual"], colnames=["Predicted"]
    )

    # Plot confusion matrix as heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt="g")
    plt.show()
    plt.clf()
    
    return confusion_matrix

def create_news_features(input_news_df, columns):
    '''Generates the percentage of news articles per month that mention a certain topic. with a rolling mean
    '''
    cols = []
    for column in columns:
        col = input_news_df.groupby(["date"])[column].mean()
        col = col.fillna(0)
        col = col.rolling(3).mean()
        col = col.shift(3)
        cols.append(col)
    return pd.concat(cols, axis=1)

def train_model_logistic(features_df, ipc_df):
    """
    Takes dataframe with keywords as input and returns the MAE, R2, train accuraccy and test accuraccy for each number of features.
    """
    ndf = ipc_df.copy()
    ndf.sort_index(level=0, inplace=True) # Sort DataFrame by date
    ndf = ndf.iloc[ndf['ipc'].notnull().argmax():].copy() # Drop rows until first notna value in ipc column
    ndf = ndf.join(features_df, how="left") # Join df with created news features
    ndf = ndf.iloc[ndf.iloc[:,-1].notnull().argmax():].copy() # Drop rows until first notna value in ipc column
    ndf.dropna(subset=ndf.iloc[:,-len(features_df.columns):].columns, inplace=True)
    ndf = ndf[ndf['ipc'] != 5]

    all_mae_values = list()
    all_r2_values = list()
    all_train_accuraccy = list()
    all_test_accuraccy = list()

    # Loops over the keywords, goes from 1 topic to all the topics
    # This still contains the 6 baseline keywords.
    for i in range(len(features_df.columns)+1):
        start = -len(features_df.columns)-6
        end = -len(features_df.columns) + i
        print(f"Start: {start}, End: {end}")
        if end == 0:
            X = ndf.iloc[:, start:] # Define explanatory variables
        else:
            X = ndf.iloc[:, start:(end)] # Define explanatory variables
        y = ndf[["ipc"]] # Define target data

        cv = TimeSeriesSplit(n_splits=5) # Define TimeSeriesSplit with 5 splits

        # Initinalize empty lists to score scores
        mae_values = list()
        r2_values = list()
        train_accuraccy = list()
        test_accuraccy = list()

        for train_index, val_index in cv.split(X): # Loop over the different training-test splits

            # Define X and y data
            X_train, X_test = X.iloc[train_index], X.iloc[val_index]
            y_train, y_test = y.iloc[train_index], y.iloc[val_index]

            # If X_train doesn't contain any news features (this happens for earlier dates) we drop news columns from both X_train and X_test
            X_train = X_train.dropna(axis=1, how='all').copy()
            X_test = X_test[X_train.columns]
            
            #Interpolate training data to generate more training points. Makes sure there are no NaN values in the training data.
            X_train = X_train.groupby('district', as_index=False).apply(lambda group: group.ffill())
            X_train.reset_index(level=0, drop=True, inplace=True)
            y_train = y_train.groupby('district', as_index=False).apply(lambda group: group.ffill())
            y_train.reset_index(level=0, drop=True, inplace=True)

            # Initialising of logistic regression model.
            model = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')

            results = model.fit(X_train, y_train.values.ravel()) # Get model results on training data

            #for looking at how the model performs closer
            #print(results.summary()) # Print model summary

            y_pred_test = results.predict(X_test) # Run model on test data
            y_pred_train = results.predict(X_train) # Run model on test data
            
            # Append results to respective lists
            mae_values.append(metrics.mean_absolute_error(y_test, y_pred_test))

            # for looking at how the model performs closer
            # plt.title('train %s - %s | test %s - %s' % (train_index[0], train_index[-1], val_index[0], val_index[-1]))
            # plot_ConfusionMatrix(prediction = y_pred_test, true = y_test['ipc']) # Plot confusion matrix

            train_accuraccy.append(metrics.accuracy_score(y_train, y_pred_train))
            test_accuraccy.append(metrics.accuracy_score(y_test, y_pred_test))
            r2_values.append(metrics.r2_score(y_train, y_pred_train))
        
        print(f"Number of features: {len(X.columns)-1}")
        print(f"Mean MAE: {np.mean(mae_values):.2f}") # Print MAE
        print(f"Mean R2: {np.mean(r2_values):.2f}") # Print R2
        print(f"Mean train accuraccy: {np.mean(train_accuraccy):.2f}") # Print train accuraccy
        print(f"Mean test accuraccy: {np.mean(test_accuraccy):.2f}") # Print test accuraccy
        print(f"Mean train accuraccy: {np.mean(train_accuraccy)}") # Print train accuraccy
        print(f"Mean test accuraccy: {np.mean(test_accuraccy)}")
        print('-------------------------')

        all_mae_values.append(np.mean(mae_values))
        all_r2_values.append(np.mean(r2_values))
        all_train_accuraccy.append(np.mean(train_accuraccy))
        all_test_accuraccy.append(np.mean(test_accuraccy))
    print("")
    print(f"All MAE values: {all_mae_values}")
    print(f"All R2 values: {all_r2_values}")
    print(f"All train accuracy values: {all_train_accuraccy}")
    print(f"All test accuracy values: {all_test_accuraccy}")
    return all_mae_values, all_r2_values, all_train_accuraccy, all_test_accuraccy

def train_model_only_articles_logistic(features_df, ipc_df):


    ndf = ipc_df.copy()
    ndf.sort_index(level=0, inplace=True) # Sort DataFrame by date
    ndf = ndf.iloc[ndf['ipc'].notnull().argmax():].copy() # Drop rows until first notna value in ipc column
    ndf = ndf.join(features_df, how="left") # Join df with created news features
    ndf = ndf.iloc[ndf.iloc[:,-1].notnull().argmax():].copy() # Drop rows until first notna value in ipc column
    ndf.dropna(subset=ndf.iloc[:,-len(features_df.columns):].columns, inplace=True)
    ndf = ndf[ndf['ipc'] != 5]

    all_mae_values = list()
    all_r2_values = list()
    all_train_accuraccy = list()
    all_test_accuraccy = list()

    for i in range(len(features_df.columns)):
        start = -len(features_df.columns)
        end = -len(features_df.columns) + i + 1
        print(f"Start: {start}, End: {end}")
        if end == 0:
            X = ndf.iloc[:, start:] # Define explanatory variables
        else:
            X = ndf.iloc[:, start:(end)] # Define explanatory variables
        y = ndf[["ipc"]] # Define target data

        cv = TimeSeriesSplit(n_splits=5) # Define TimeSeriesSplit with 5 splits

        # Initinalize empty lists to score scores
        mae_values = list()
        r2_values = list()
        train_accuraccy = list()
        test_accuraccy = list()

        for train_index, val_index in cv.split(X): # Loop over the different training-test splits

            # Define X and y data
            X_train, X_test = X.iloc[train_index], X.iloc[val_index]
            y_train, y_test = y.iloc[train_index], y.iloc[val_index]

            # If X_train doesn't contain any news features (this happens for earlier dates) we drop news columns from both X_train and X_test
            X_train = X_train.dropna(axis=1, how='all').copy()
            X_test = X_test[X_train.columns]
            
            #Interpolate training data to generate more training points
            X_train = X_train.groupby('district', as_index=False).apply(lambda group: group.ffill())
            X_train.reset_index(level=0, drop=True, inplace=True)
            y_train = y_train.groupby('district', as_index=False).apply(lambda group: group.ffill())
            y_train.reset_index(level=0, drop=True, inplace=True)

            model = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')

            results = model.fit(X_train, y_train.values.ravel()) # Get model results on training data

            #for looking at how the model performs closer
            #print(results.summary()) # Print model summary

            y_pred_test = results.predict(X_test) # Run model on test data
            y_pred_train = results.predict(X_train) # Run model on test data
            
            # Append results to respective lists
            mae_values.append(metrics.mean_absolute_error(y_test, y_pred_test))
            r2_values.append(metrics.r2_score(y_train, y_pred_train))
            # r2_values.append(results.rsquared)

            # for looking at how the model performs closer
            # plt.title('train %s - %s | test %s - %s' % (train_index[0], train_index[-1], val_index[0], val_index[-1]))
            # plot_ConfusionMatrix(prediction = y_pred_test, true = y_test['ipc']) # Plot confusion matrix
        
            # Append accuraccy to respective lists
            train_accuraccy.append(metrics.accuracy_score(y_train, y_pred_train))
            test_accuraccy.append(metrics.accuracy_score(y_test, y_pred_test))
        
        print(f"Number of features: {len(X.columns)-1}")
        print(f"Mean MAE: {np.mean(mae_values):.2f}") # Print MAE
        print(f"Mean R2: {np.mean(r2_values):.2f}") # Print R2
        print(f"Mean train accuraccy: {np.mean(train_accuraccy):.2f}") # Print train accuraccy
        print(f"Mean test accuraccy: {np.mean(test_accuraccy):.2f}") # Print test accuraccy
        print(f"Mean train accuraccy: {np.mean(train_accuraccy)}") # Print train accuraccy
        print(f"Mean test accuraccy: {np.mean(test_accuraccy)}")
        print('-------------------------')

        all_mae_values.append(np.mean(mae_values))
        all_r2_values.append(np.mean(r2_values))
        all_train_accuraccy.append(np.mean(train_accuraccy))
        all_test_accuraccy.append(np.mean(test_accuraccy))
    print("")
    print(f"All MAE values: {all_mae_values}")
    print(f"All R2 values: {all_r2_values}")
    print(f"All train accuracy values: {all_train_accuraccy}")
    print(f"All test accuracy values: {all_test_accuraccy}")
    return all_mae_values, all_r2_values, all_train_accuraccy, all_test_accuraccy

def train_model_only_articles_OLS(features_df, ipc_df):


    ndf = ipc_df.copy()
    ndf.sort_index(level=0, inplace=True) # Sort DataFrame by date
    ndf = ndf.iloc[ndf['ipc'].notnull().argmax():].copy() # Drop rows until first notna value in ipc column
    ndf = ndf.join(features_df, how="left") # Join df with created news features
    ndf = ndf.iloc[ndf.iloc[:,-1].notnull().argmax():].copy() # Drop rows until first notna value in ipc column
    ndf.dropna(subset=ndf.iloc[:,-len(features_df.columns):].columns, inplace=True)
    ndf = ndf[ndf['ipc'] != 5]

    all_mae_values = list()
    all_r2_values = list()
    all_train_accuraccy = list()
    all_test_accuraccy = list()

    for i in range(len(features_df.columns)):
        start = -len(features_df.columns)
        end = -len(features_df.columns) + i + 1
        print(f"Start: {start}, End: {end}")
        if end == 0:
            X = ndf.iloc[:, start:] # Define explanatory variables
        else:
            X = ndf.iloc[:, start:(end)] # Define explanatory variables
        y = ndf[["ipc"]] # Define target data

        cv = TimeSeriesSplit(n_splits=5) # Define TimeSeriesSplit with 5 splits

        # Initinalize empty lists to score scores
        mae_values = list()
        r2_values = list()
        train_accuraccy = list()
        test_accuraccy = list()

        for train_index, val_index in cv.split(X): # Loop over the different training-test splits

            # Define X and y data
            X_train, X_test = X.iloc[train_index], X.iloc[val_index]
            y_train, y_test = y.iloc[train_index], y.iloc[val_index]

            # If X_train doesn't contain any news features (this happens for earlier dates) we drop news columns from both X_train and X_test
            X_train = X_train.dropna(axis=1, how='all').copy()
            X_test = X_test[X_train.columns]
            
            #Interpolate training data to generate more training points
            X_train = X_train.groupby('district', as_index=False).apply(lambda group: group.ffill())
            X_train.reset_index(level=0, drop=True, inplace=True)
            y_train = y_train.groupby('district', as_index=False).apply(lambda group: group.ffill())
            y_train.reset_index(level=0, drop=True, inplace=True)

            model = OLS(y_train, X_train, missing="drop")

            results = model.fit() # Get model results on training data
            #print(results.summary()) # Print model summary

            y_pred_test = results.predict(X_test) # Run model on test data
            y_pred_train = results.predict(X_train) # Run model on test data
            
            # Append results to respective lists
            mae_values.append(metrics.mean_absolute_error(y_test, y_pred_test))
            r2_values.append(metrics.r2_score(y_train, y_pred_train))
            # r2_values.append(results.rsquared)

            # plt.title('train %s - %s | test %s - %s' % (train_index[0], train_index[-1], val_index[0], val_index[-1]))
            # plot_ConfusionMatrix(prediction = y_pred_test, true = y_test['ipc']) # Plot confusion matrix

            # Calculate accuraccy
            train_correct = ((y_pred_train.round()==y_train['ipc']).sum()) # ammount of correct train predictions
            test_correct = ((y_pred_test.round()==y_test['ipc']).sum()) # ammount of correct test predictions
            train_values_count = len(y_train['ipc'])-y_pred_train.round().isna().sum() # train values contain NaNs, so to calculate what values have been predicted we need to subtract the number of NaNs from the total number of values
            test_values_count = len(y_test['ipc'])-y_test['ipc'].isna().sum() # test values contain NaNs, so to calculate what values have been predicted we need to subtract the number of NaNs from the total number of values
        
            # Append accuraccy to respective lists
            train_accuraccy.append(train_correct/train_values_count)
            test_accuraccy.append(test_correct/test_values_count)

        
        print(f"Number of features: {len(X.columns)-1}")
        print(f"Mean MAE: {np.mean(mae_values):.2f}") # Print MAE
        print(f"Mean R2: {np.mean(r2_values):.2f}") # Print R2
        print(f"Mean train accuraccy: {np.mean(train_accuraccy):.2f}") # Print train accuraccy
        print(f"Mean test accuraccy: {np.mean(test_accuraccy):.2f}") # Print test accuraccy
        print(f"Mean train accuraccy: {np.mean(train_accuraccy)}") # Print train accuraccy
        print(f"Mean test accuraccy: {np.mean(test_accuraccy)}")
        print('-------------------------')

        all_mae_values.append(np.mean(mae_values))
        all_r2_values.append(np.mean(r2_values))
        all_train_accuraccy.append(np.mean(train_accuraccy))
        all_test_accuraccy.append(np.mean(test_accuraccy))
    print("")
    print(f"All MAE values: {all_mae_values}")
    print(f"All R2 values: {all_r2_values}")
    print(f"All train accuracy values: {all_train_accuraccy}")
    print(f"All test accuracy values: {all_test_accuraccy}")
    return all_mae_values, all_r2_values, all_train_accuraccy, all_test_accuraccy

def train_model_OLS(features_df, ipc_df):


    ndf = ipc_df.copy()
    ndf.sort_index(level=0, inplace=True) # Sort DataFrame by date
    ndf = ndf.iloc[ndf['ipc'].notnull().argmax():].copy() # Drop rows until first notna value in ipc column
    ndf = ndf.join(features_df, how="left") # Join df with created news features
    ndf = ndf.iloc[ndf.iloc[:,-1].notnull().argmax():].copy() # Drop rows until first notna value in ipc column
    ndf.dropna(subset=ndf.iloc[:,-len(features_df.columns):].columns, inplace=True)
    ndf = ndf[ndf['ipc'] != 5]

    all_mae_values = list()
    all_r2_values = list()
    all_train_accuraccy = list()
    all_test_accuraccy = list()

    for i in range(len(features_df.columns)+1):  # +1 for 0 topics
        start = -len(features_df.columns) - 6  # -6 is for baseline features.
        end = -len(features_df.columns) + i
        print(f"Start: {start}, End: {end}")
        if end == 0:
            X = ndf.iloc[:, start:] # Define explanatory variables
        else:
            X = ndf.iloc[:, start:(end)] # Define explanatory variables
        y = ndf[["ipc"]] # Define target data

        cv = TimeSeriesSplit(n_splits=5) # Define TimeSeriesSplit with 5 splits

        # Initinalize empty lists to score scores
        mae_values = list()
        r2_values = list()
        train_accuraccy = list()
        test_accuraccy = list()

        for train_index, val_index in cv.split(X): # Loop over the different training-test splits

            # Define X and y data
            X_train, X_test = X.iloc[train_index], X.iloc[val_index]
            y_train, y_test = y.iloc[train_index], y.iloc[val_index]

            # If X_train doesn't contain any news features (this happens for earlier dates) we drop news columns from both X_train and X_test
            X_train = X_train.dropna(axis=1, how='all').copy()
            X_test = X_test[X_train.columns]
            
            #Interpolate training data to generate more training points
            X_train = X_train.groupby('district', as_index=False).apply(lambda group: group.ffill())
            X_train.reset_index(level=0, drop=True, inplace=True)
            y_train = y_train.groupby('district', as_index=False).apply(lambda group: group.ffill())
            y_train.reset_index(level=0, drop=True, inplace=True)

            model = OLS(y_train, X_train, missing="drop")

            results = model.fit() # Get model results on training data
            #print(results.summary()) # Print model summary

            y_pred_test = results.predict(X_test) # Run model on test data
            y_pred_train = results.predict(X_train) # Run model on test data
            
            # Append results to respective lists
            mae_values.append(metrics.mean_absolute_error(y_test, y_pred_test))
            # r2_values.append(results.rsquared)
            r2_values.append(metrics.r2_score(y_train, y_pred_train))

            # plt.title('train %s - %s | test %s - %s' % (train_index[0], train_index[-1], val_index[0], val_index[-1]))
            # plot_ConfusionMatrix(prediction = y_pred_test, true = y_test['ipc']) # Plot confusion matrix

            # Calculate accuraccy
            train_correct = ((y_pred_train.round()==y_train['ipc']).sum()) # ammount of correct train predictions
            test_correct = ((y_pred_test.round()==y_test['ipc']).sum()) # ammount of correct test predictions
            train_values_count = len(y_train['ipc'])-y_pred_train.round().isna().sum() # train values contain NaNs, so to calculate what values have been predicted we need to subtract the number of NaNs from the total number of values
            test_values_count = len(y_test['ipc'])-y_test['ipc'].isna().sum() # test values contain NaNs, so to calculate what values have been predicted we need to subtract the number of NaNs from the total number of values
        
            # Append accuraccy to respective lists
            train_accuraccy.append(train_correct/train_values_count)
            test_accuraccy.append(test_correct/test_values_count)

        
        print(f"Number of features: {len(X.columns)-1}")
        print(f"Mean MAE: {np.mean(mae_values):.2f}") # Print MAE
        print(f"Mean R2: {np.mean(r2_values):.2f}") # Print R2
        print(f"Mean train accuraccy: {np.mean(train_accuraccy):.2f}") # Print train accuraccy
        print(f"Mean test accuraccy: {np.mean(test_accuraccy):.2f}") # Print test accuraccy
        print(f"Mean train accuraccy: {np.mean(train_accuraccy)}") # Print train accuraccy
        print(f"Mean test accuraccy: {np.mean(test_accuraccy)}")
        print('-------------------------')

        all_mae_values.append(np.mean(mae_values))
        all_r2_values.append(np.mean(r2_values))
        all_train_accuraccy.append(np.mean(train_accuraccy))
        all_test_accuraccy.append(np.mean(test_accuraccy))
    print("")
    print(f"All MAE values: {all_mae_values}")
    print(f"All R2 values: {all_r2_values}")
    print(f"All train accuracy values: {all_train_accuraccy}")
    print(f"All test accuracy values: {all_test_accuraccy}")
    return all_mae_values, all_r2_values, all_train_accuraccy, all_test_accuraccy

def extract_rake_summary_keywords(df):
    keywords_list = []
    for index, row in df.iterrows():
        for item in row['summary_rake_keywords']:
            keywords_list.append(item)

    counts = Counter(keywords_list)

    keywords_rake_summary = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    keywords_rake_summary.rename(columns={0: 'values'}, inplace=True)
    keywords_rake_summary = keywords_rake_summary.sort_values(by='values', ascending=False)

    return keywords_rake_summary

def extract_rake_paragraphs_keywords(df):
    keywords_list = []
    for index, row in df.iterrows():
        for item in row['paragraphs_rake_keywords']:
            keywords_list.append(item)

    # Count the keywords and sort them using the Counter library
    counts = Counter(keywords_list)

    # Create a dataframe to sort the keywords more easily
    keywords_rake_paragraphs = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    keywords_rake_paragraphs.rename( columns={0 :'values'}, inplace=True )
    keywords_rake_paragraphs.sort_values(by='values', ascending=False)
    return keywords_rake_paragraphs